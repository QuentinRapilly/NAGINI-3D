import yaml
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import wandb
import numpy as np
import random
from os import environ
from os.path import join, basename, normpath
from argparse import ArgumentParser

from nagini3D.data.training_set import TrainingSet, custom_collate
from nagini3D.data.th_optim_set import OptimSet
from nagini3D.models.model import Nagini3D

# reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def retrain(path_to_model_dir, checkpoint_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(join(path_to_model_dir, "config.yaml"), 'r') as file:
        cfg = yaml.safe_load(file)

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

    init_sphere = cfg["settings"]["init_sphere"]

    min_cell_ratio = cfg["data"].get("min_cell_ratio")
    if min_cell_ratio is None:
        min_cell_ratio = 0

    anisotropy_values = cfg["data"].get("anisotropy")
    if anisotropy_values is None:
        anisotropy = None
    else:
        anisotropy = [float(x) for x in anisotropy_values.split(",")]

    intensity_aug = cfg["data"].get("intensity_aug")
    if intensity_aug == None:
        intensity_aug = []
    else:
        intensity_aug = intensity_aug.split(",")


    r_mean = cfg["settings"]["r_mean"]
    train_set = TrainingSet(nb_points=P, r_mean=r_mean, img_size=cfg["data"]["img_size"],
                            data_aug=cfg["data"]["data_aug"], dataset_dir=cfg["data"]["train"],
                            max_noise=cfg["data"]["max_noise"], binary_img=cfg["data"]["binary_img"],
                            anisotropy_ratio=anisotropy, cell_ratio_th=min_cell_ratio,
                            intensity_aug=intensity_aug)
    val_set = TrainingSet(nb_points=P, r_mean=train_set.r_mean,  img_size=cfg["data"]["img_size"],
                            data_aug=cfg["data"]["data_aug"], dataset_dir=cfg["data"]["val"],
                            max_noise=cfg["data"]["max_noise"], binary_img=cfg["data"]["binary_img"],
                            anisotropy_ratio=anisotropy, cell_ratio_th=min_cell_ratio,
                            intensity_aug=intensity_aug)


    print(f"Mean radius of cells : {round(train_set.r_mean,3)}")

    train_loader = DataLoader(train_set, collate_fn=custom_collate, batch_size=cfg["settings"]["batch_size"],\
                              worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_set, collate_fn=custom_collate, batch_size=cfg["settings"]["batch_size"],\
                            worker_init_fn=seed_worker, generator=g)

    nb_epochs = cfg["settings"]["nb_epochs"]

    use_wandb = cfg["settings"]["use_wandb"]


    M1 = cfg["settings"]["M1"]
    M2 = cfg["settings"]["M2"]

    use_scale = cfg["settings"].get("use_scale")
    if use_scale == None: use_scale = True

    model = Nagini3D(unet_cfg=cfg["model"], P=P, M1=M1, M2=M2, device=device,
                     use_scale=use_scale, save_path=cfg["save"]["path"])
    

    first_epoch, wandb_id, previous_val_loss = model.load_checkpoint(join(path_to_model_dir, checkpoint_name),
                                                                     optimizer_cfg=cfg["optimizer"])
    
    
    model.init_loss(loss_lambdas=cfg["loss"]["lambdas"], reg_ratio=cfg["loss"]["reg_ratio"])
    model.reload_save_dir(basename(normpath(path_to_model_dir)))

    cfg["settings"] = {**cfg["settings"], "r_mean" : train_set.r_mean}

    model.save_config(cfg_dict=cfg)

    best_val_score = previous_val_loss
    epoch_best_val = first_epoch

    if use_wandb:
        run = wandb.init(project="SplineDist3D_dense", id=wandb_id, resume="allow")

    for epoch_idx in range(first_epoch+1, nb_epochs):
        print(f"Epoch {epoch_idx+1} / {nb_epochs}\nTraining ...")

        #snake_ratio = model.update_snake_loss(epoch_idx)
        reg_ratio = 1 #- snake_ratio

        loss = model.epoch(data_loader = train_loader, init_with_sphere=init_sphere)

        print(f"\nLoss : {loss['loss']}\nTesting ...")

        validation, cloud_list = model.val(data_loader = val_loader, init_with_sphere=init_sphere, nb_cells_to_plot=4)
        
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

        #model.scheduler_step(validation["loss"])

        if validation["loss"] < best_val_score:
            model.save_model("best", epoch=epoch_idx, wandb_id=run.id, val_loss = validation["loss"])
            best_val_score = validation["loss"]
            epoch_best_val = epoch_idx


        
    cfg["save"] = {**cfg["save"], "best_epoch" : epoch_best_val}
    model.save_config(cfg_dict=cfg)
    model.save_model("final",  epoch=epoch_idx, wandb_id=run.id, val_loss = validation["loss"])

    wandb.finish()

    model.load_weights(join(model.save_dir, "best.pkl"))
    optim_set = OptimSet(cfg["data"]["val"])
    opti_th = model.optimize_thresholds(optim_set, r_mean=train_set.r_mean, nb_tiles=(1,1,1))
    model.save_th(th_dict=opti_th)

    return model, train_set.r_mean


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-n", "--name", default="best_chackpoint.pkl")
    args = parser.parse_args()
    retrain(args.input, args.name)