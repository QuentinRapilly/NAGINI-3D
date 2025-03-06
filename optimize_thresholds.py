from argparse import ArgumentParser
import yaml
import torch
import tifffile
from os.path import join, basename, splitext, isdir, normpath
from os import mkdir
from glob import glob
from numpy import savez
from omegaconf import OmegaConf

from nagini3D.models.model import DeepBioSnake
from nagini3D.data.th_optim_set import OptimSet


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", help="str: path to the set dir")
    parser.add_argument("-m", "--model", help="str: path to the model config and weights directory")
    parser.add_argument("-w", "--weights", default="best.pkl", help="str: precise the name of the weights file (only name, not path) if you want a different one from best epoch")
    parser.add_argument("-t", "--tiles", default="1,1,1", help="(int,int,int): number of slices along each dimensions (ex : 1,1,1), if 'cuda out of memory' occurs")

    args = parser.parse_args()

    input_path = args.input
    model_dir = args.model 
    weigth_file = args.weights
    nb_tiles = [int(x) for x in args.tiles.split(",")]

    model_name = basename(normpath(model_dir))


    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(join(model_dir, "config.yaml"), 'r') as file:
        cfg = yaml.safe_load(file)

    M1 = cfg["settings"]["M1"]
    M2 = cfg["settings"]["M2"]
    r_mean = cfg["settings"]["r_mean"]

    weight_path = join(model_dir, weigth_file)

    model = DeepBioSnake(unet_cfg=cfg["model"], P = 101, M1 = M1, M2 = M2, save_path=".", device=device)
    model.load_weights(weight_path)

    optim_set = OptimSet(input_path)

    ths = model.optimize_thresholds(optim_set=optim_set, r_mean=r_mean, nb_tiles=nb_tiles, maxiter=10)

    OmegaConf.save(config=ths, f=join(model_dir, "thresholds.yaml"))