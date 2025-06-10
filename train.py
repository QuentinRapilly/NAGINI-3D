import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import wandb
import numpy as np
import random
from os import environ
from os.path import join

from nagini3D.data.training_set import TrainingSet, custom_collate
from nagini3D.data.th_optim_set import OptimSet
from nagini3D.models.model import Nagini3D

# reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def start_training(cfg : DictConfig):
    train(cfg)

def train(cfg : DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # reproducibility
    fixed_seed = cfg["settings"]["seed"]
    g = torch.Generator()
    if type(fixed_seed) == int:
        print(f"Selected seed : {fixed_seed}")
        environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
        torch.manual_seed(fixed_seed)
        random.seed(fixed_seed)
        np.random.seed(fixed_seed)
        torch.use_deterministic_algorithms(True)
        g.manual_seed(fixed_seed)
        

    P = cfg["settings"]["nb_points_on_surface"] # number of points to sample for each snake

    min_cell_ratio = cfg["data"].get("min_cell_ratio")
    if min_cell_ratio is None:
        min_cell_ratio = 0

    patch_size = [int(x) for x in cfg["data"]["patch_size"]]

    anisotropy = [float(x) for x in cfg["data"]["anisotropy"].split(",")]


    train_set = TrainingSet(nb_points=P, patch_size=patch_size, dataset_dir=cfg["data"]["train"],
                            data_aug=cfg["data"]["data_aug"], cell_ratio_th=min_cell_ratio, anisotropy_ratio=anisotropy)
    val_set = TrainingSet(nb_points=P, r_mean=train_set.r_mean,  patch_size=patch_size,\
                            data_aug=cfg["data"]["data_aug"], dataset_dir=cfg["data"]["val"],
                            cell_ratio_th=min_cell_ratio, anisotropy_ratio=anisotropy)


    print(f"Mean radius of cells : {round(train_set.r_mean,3)}")

    train_loader = DataLoader(train_set, collate_fn=custom_collate, batch_size=cfg["settings"]["batch_size"],\
                              worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_set, collate_fn=custom_collate, batch_size=cfg["settings"]["batch_size"],\
                            worker_init_fn=seed_worker, generator=g)

    nb_epochs = cfg["settings"]["nb_epochs"]

    use_wandb = cfg["settings"]["use_wandb"]


    M1 = cfg["settings"]["M1"]
    M2 = cfg["settings"]["M2"]

    model = Nagini3D(unet_cfg=cfg["model"], P=P, M1=M1, M2=M2,\
                         device=device, save_path=cfg["save"]["path"])
    
    model.init_optimizer(optimizer_cfg=cfg["optimizer"])
    #model.init_scheduler(scheduler_cfg=cfg["scheduler"])
    
    model.init_loss(loss_lambdas=cfg["loss"]["lambdas"], snake_loss=cfg["loss"]["snake_loss"],
                    reg_part=cfg["loss"]["reg_part"], epoch_reg_max=cfg["loss"]["epoch_reg_max"])
    model.set_name_and_save_dir(cfg["settings"]["experiment_name"])

    cfg["settings"] = {**cfg["settings"], "r_mean" : train_set.r_mean}

    model.save_config(cfg_dict=cfg)

    best_val_score = float("inf")
    epoch_best_val = -1

    if use_wandb:
        run = wandb.init(project="SplineDist3D_dense", name=model.get_run_name(), config=dict(cfg))

    for epoch_idx in range(nb_epochs):
        print(f"Epoch {epoch_idx+1} / {nb_epochs}\nTraining ...")

        snake_ratio = model.update_snake_loss(epoch_idx)
        reg_ratio = 1 - snake_ratio

        loss = model.epoch(data_loader = train_loader)

        print(f"\nLoss : {loss['loss']}\nTesting ...")

        validation, cloud_list = model.val(data_loader = val_loader, nb_cells_to_plot=4)
        
        print(f"\nAccuracy on validation set : {validation['loss']}")



        if use_wandb:
            wandb.log({
                "train/snakes" : loss["snakes"], "train/spots" : loss["spots"], "train/loss" : loss["loss"],\
                    "train/reg" : loss["regularization"], "train/variance" : loss["fm_var"],\
                        "test/snakes" : validation["snakes"], "test/spots" : validation["spots"], "test/loss" : validation["loss"],\
                            "test/reg" : validation["regularization"], "test/variance" : validation["fm_var"],\
                                "lr" : model.optimizer.param_groups[0]['lr'], "reg_part" : reg_ratio,\
                "points_clouds" : [
                    wandb.Object3D(point_cloud) for point_cloud in cloud_list
                ]
            })


        if validation["loss"] < best_val_score:
            model.save_model(f"best")
            best_val_score = validation["loss"]
            epoch_best_val = epoch_idx

        
    cfg["save"] = {**cfg["save"], "best_epoch" : epoch_best_val}
    model.save_config(cfg_dict=cfg)
    model.save_model("final")

    wandb.finish()

    model.load_weights(join(model.save_dir, "best.pkl"))
    optim_set = OptimSet(cfg["data"]["val"])
    opti_th = model.optimize_thresholds(optim_set, r_mean=train_set.r_mean, nb_tiles=(1,1,1))
    model.save_th(th_dict=opti_th)

    return model, train_set.r_mean


if __name__ == "__main__":
    start_training()