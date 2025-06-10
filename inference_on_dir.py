from argparse import ArgumentParser,ArgumentTypeError
import yaml
import torch
import tifffile
from os.path import join, basename, splitext, isdir, normpath
from os import mkdir
from glob import glob
from numpy import savez

from nagini3D.models.model import Nagini3D


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", help="str: path to the image directory")
    parser.add_argument("-m", "--model", help="str: path to the model config and weights directory")
    parser.add_argument("-w", "--weights", default="best.pkl", help="str: precise the name of the weights file (only name, not path) if you want a different one from best epoch")
    parser.add_argument("-o", "--output", help="str: path to the folder where to store inference results")
    parser.add_argument("-t", "--tiles", default="1,1,1", help="(int,int,int): number of splits along each dimensions (ex : 1,1,1), if the process exceed memory")
    parser.add_argument("-tt", "--thresholds", default="none", help="(float,float): precise the prob and nms thresholds to use if you don't want to appply the optimized ones (ex : 0.5,0.5)")
    parser.add_argument("-s", "--snake", type=str2bool, default=True, help="bool: weither to apply the snake refinement step")
    parser.add_argument("-ot", "--otsu", type=str2bool, default=True, help="bool: weither to apply otsu thresholding on image before snake refinement, more precise for sparse objects, less precise for dense objects.")
    parser.add_argument("-a", "--anisotropy", type=str, default="1,1,1", help="float,float,float: anisotrpoy ratio along each axis")

    args = parser.parse_args()

    input_path = args.input
    model_dir = args.model 
    weigth_file = args.weights
    result_path = args.output
    nb_tiles = [int(x) for x in args.tiles.split(",")]

    use_snake = args.snake
    use_otsu = args.otsu

    anisotropy = [float(x) for x in args.anisotropy.split(",")]

    model_name = basename(normpath(model_dir))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(join(model_dir, "config.yaml"), 'r') as file:
        cfg = yaml.safe_load(file)

    if args.thresholds == "none":
        with open(join(model_dir, "thresholds.yaml"), 'r') as file:
            thresholds = yaml.safe_load(file)
    else:
        thresholds = [float(x) for x in args.thresholds.split(",")]
        thresholds = {"prob" : thresholds[0], "nms" : thresholds[1]}

    M1 = cfg["settings"]["M1"]
    M2 = cfg["settings"]["M2"]
    r_mean = cfg["settings"]["r_mean"]

    weight_path = join(model_dir, weigth_file)

    model = Nagini3D(unet_cfg=cfg["model"], P = 101, M1 = M1, M2 = M2, save_path=result_path, device=device)
    model.load_weights(weight_path)

    imgs_path = glob(join(input_path,"*.tif"))

    proba_th = thresholds["prob"] 
    nms_th = thresholds["nms"]

    save_dir = join(result_path, model_name+f"_th_{round(proba_th,3)}_nms_{nms_th}")
    if not isdir(save_dir):
        mkdir(save_dir)

    mask_dir = join(save_dir, "mask")
    if not isdir(mask_dir): mkdir(mask_dir)
    

    for img_path in imgs_path:

        if device == "cuda":
            torch.cuda.empty_cache()

        img = tifffile.imread(img_path)
        img_name = basename(img_path)
        name = splitext(img_name)[0]

        print(f"Processing image : {img_name}")

        mask, proba, parameters, surfaces = model.inference(img, proba_th=proba_th, r_mean=r_mean, nb_tiles=nb_tiles,\
                                              nms_th=nms_th, optim_snake=use_snake, otsu=use_otsu, anisotropy=anisotropy)


        mask_path = join(result_path, img_name)
        proba_path = join(result_path, "proba_"+img_name)
        surfaces_path = join(result_path, "surf_"+name+".npz")
        tifffile.imwrite(mask_path, mask.cpu().numpy())
        tifffile.imwrite(proba_path, proba.cpu().numpy())
        savez(surfaces_path, points = surfaces["points"], facets = surfaces["facets"], values = surfaces["values"])