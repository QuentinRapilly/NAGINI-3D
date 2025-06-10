import argparse
from glob import glob
from os.path import join, splitext, basename
import tifffile
import numpy as np
from time import time
from torch.cuda import is_available


from nagini3D.data.format_dataset.data_reading_tools import (compute_barycenter, compute_radius, mask_to_contour,
                                bound_box, farthest_point_sampling, distance_to_center)


#PROBA_TYPE = "distance"
PROBA_TYPE = "barycentric"

#SAMPLING_TYPE = "erosion"
SAMPLING_TYPE = "dilation"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="str: direrctory where the segmentation masks are stored")
    parser.add_argument("-o", "--output", help="str: directory where to store the formated sampling")
    parser.add_argument("-n", "--nb_sampling", type=int, default=101,
                        help="int: number of points to sample on mask boundaries")
    parser.add_argument("-s", "--sampling", type=str2bool, nargs="?", default=True, const=True, help="bool: weither the sampling step should be done or not")
    parser.add_argument("-p", "--proba", type=str2bool, nargs="?", default=True, const=True, help="bool: weither the probability step should be done or not")
    parser.add_argument("-v", "--verbose", type=str2bool, nargs="?", default=False, const=True, help="bool: weither to print logs of the sampling procedure")
    parser.add_argument("-a", "--anisotropy", default="1,1,1", help="str: If your images are strongly anisotropic, precise the anisotropy ratio between axis (with float greater or equal than 1, separated by commas, ex:1,3,1)")

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    nb_sampling = args.nb_sampling
    do_sample = args.sampling
    do_proba = args.proba
    verbose = args.verbose
    anisotropy = [float(x) for x in args.anisotropy.split(",")]

    device = "cuda" if is_available() else "cpu"

    mask_files = glob(join(input_dir,"*.tif"))

    for mask_f in mask_files: 
        tic = time()
        filename = basename(mask_f)
        name_no_ext = splitext(filename)[0]
        if verbose: print(f"Processing file : {filename}\n")

        mask = tifffile.imread(mask_f)

        nx,ny,nz = mask.shape
        vx,vy,vz = np.arange(nx)[:,None,None], np.arange(ny)[None,:,None], np.arange(nz)[None, None, :]
        mesh = (vx, vy, vz)

        #mesh = np.meshgrid(vx, vy, vz)

        mask_idx = np.unique(mask)[1:]

        nb_cells = len(mask_idx)

        proba_map = np.zeros_like(mask, dtype=float)
        gaussian_mask = np.zeros_like(mask, dtype=int)

        center_list = list()
        radius_list = list()
        sampling_list = list()
        idx_list = list()

        cells_count = 0

        for i,idx in enumerate(mask_idx):
            crt_mask = (mask == idx)*1
            N = crt_mask.sum()
            if verbose: print(f"\rCell nb {idx}/{nb_cells}", end="")
        
            barycenter = compute_barycenter(crt_mask, mesh)

            crt_contour = mask_to_contour(crt_mask, mode = SAMPLING_TYPE)

            radius = compute_radius(crt_contour, mesh, barycenter)

            bb = bound_box(crt_contour, mesh)

            x_min, x_max, y_min, y_max, z_min, z_max = bb

            small_contour = crt_contour[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
            small_mask = crt_mask[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

            if do_proba:
                if PROBA_TYPE == "distance":
                    distance_map = distance_to_center(mask=small_mask)
                    proba_map[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = np.maximum(distance_map, proba_map[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1])
                

                if PROBA_TYPE=="barycentric":
                    bx, by, bz = barycenter
                    dist = np.sqrt((vx-bx)**2+(vy-by)**2+(vz-bz)**2)
                    masked_dist =  dist*crt_mask
                    M = np.max(masked_dist)
                    masked_dist = (M - masked_dist)*crt_mask
                    den = np.max(masked_dist)
                    if den>0:
                        masked_dist = masked_dist/den
                        proba_map += masked_dist

            if do_sample:
                sampling = farthest_point_sampling(small_contour, nb_sampling, device = device, anisotropy=anisotropy)
                centered_sampling = np.array(sampling) + np.array([x_min, y_min, z_min]) - np.array([barycenter])
                sampling_list.append(centered_sampling)

            center_list.append(barycenter)
            radius_list.append(radius)
            idx_list.append(cells_count)

            cells_count+=1


        tac = time()

        if verbose: print(f"\nTime spent to process file : {round(tac-tic)}s")

        if do_sample:
            npz_path = join(output_dir, name_no_ext+".npz")
            np.savez(npz_path, centers = np.array(center_list), radius = np.array(radius_list),\
                 samplings = np.array(sampling_list))
        
        if do_proba:
            proba_path = join(output_dir, name_no_ext+".tif")
            tifffile.imwrite(proba_path, proba_map)
