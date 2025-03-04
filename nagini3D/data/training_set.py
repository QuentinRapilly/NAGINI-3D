from torch.utils.data import Dataset
from os.path import join, basename, splitext
from glob import glob
import tifffile
import numpy as np
from random import randint
import torch
from typing import Any
from csbdeep.utils import normalize

from .data_aug_tools import generate_rotation_matrix, rotate_image

FLOAT_TYPE = np.float32
INT_TYPE = np.uint16

class TrainingSet(Dataset):

    def __init__(self, dataset_dir : str, patch_size : int, nb_points : int = 101, anisotropy_ratio : list = [1,1,1],
                 r_mean = None, data_aug : bool = True, cell_ratio_th = 0.0, **kwargs) -> None:
        super().__init__()

        self.images_dir = join(dataset_dir, "images")
        self.samplings_dir = join(dataset_dir, "samplings")
        self.masks_dir = join(dataset_dir, "masks")

        self.patch_size = patch_size
        self.nb_points = nb_points
        self.data_aug = data_aug
        self.anisotropy_ratio = np.array([[anisotropy_ratio]])

        self.imgs_list = sorted(glob(join(self.images_dir, "*.tif")))

        self.samplings_list = glob(join(self.samplings_dir, "*.npz"))

        self.cell_ratio_th = cell_ratio_th


        if r_mean != None:
            self.r_mean = r_mean

        else:
            r_mean = 0
            nb_cells = 0
            for sampling_file in self.samplings_list:
                radius = np.load(sampling_file)["radius"]
                n = len(radius)
                r_mean = nb_cells/(n+nb_cells)*r_mean + n/(n+nb_cells)*radius.mean()
                nb_cells += n

            self.nb_cells = nb_cells
            self.r_mean = float(r_mean)



    def __len__(self):
        return len(self.imgs_list)
    

    def crop_image(self, img) -> tuple:
        nx,ny,nz = img.shape 
        M = self.patch_size
        x_max, y_max, z_max = nx-M, ny-M, nz-M
        x, y, z = randint(0, x_max), randint(0, y_max), randint(0, z_max)
        return img[x:x+M, y:y+M, z:z+M], (x,y,z)

    def __getitem__(self, index) -> Any:
        img_path = self.imgs_list[index]
        img_name = basename(img_path)
        img_name_no_ext = splitext(img_name)[0]


        full_img = tifffile.imread(img_path)
        full_img = normalize(full_img, pmin=1, pmax=99.8, axis=(0,1,2), clip=True)

        full_proba = tifffile.imread(join(self.samplings_dir, img_name))
        full_mask = tifffile.imread(join(self.masks_dir, img_name))

        samplings_arrays = np.load(join(self.samplings_dir, img_name_no_ext+".npz"))

        full_samplings = samplings_arrays["samplings"]
        full_centers = samplings_arrays["centers"]

        M = self.patch_size

        mask, (x,y,z) = self.crop_image(full_mask)
        mask = mask.astype(INT_TYPE)

        if self.cell_ratio_th != 0: # avoid to compute the sum when one chooses the ratio equal to 0 (i.e. no verification)
            cell_ratio_in_mask = (mask>0).sum()/M**3
            while cell_ratio_in_mask < self.cell_ratio_th:
                mask, (x,y,z) = self.crop_image(full_mask)
                mask = mask.astype(INT_TYPE)
                cell_ratio_in_mask = (mask>0).sum()/M**3


        proba = full_proba[x:x+M, y:y+M, z:z+M].astype(FLOAT_TYPE)
        img = full_img[x:x+M, y:y+M, z:z+M].astype(FLOAT_TYPE)

        unique_lbls = np.unique(full_mask)[1:]
        lbls = np.arange(unique_lbls[-1]+1, dtype=int)
        lbls[unique_lbls] = np.arange(len(unique_lbls), dtype=int)


        if self.data_aug:
            theta1, theta2, theta3 = randint(0,3), randint(0,3), randint(0,3)
            angles = (theta1, theta2, theta3)

            img = rotate_image(img, angles).copy()
            proba = rotate_image(proba, angles).copy()
            mask = rotate_image(mask, angles).copy()
        
        S = self.patch_size/np.array(mask.shape)

        cell_voxels = mask.nonzero()
        voxels_coordinates = np.stack(cell_voxels).T
        voxels_idx = lbls[mask[cell_voxels]]
        voxels_proba = proba[cell_voxels]
        voxels_barycenters = (full_centers[voxels_idx] - np.array([x,y,z]))

        assert self.nb_points <= full_samplings.shape[1]
        tmp_samplings = full_samplings[voxels_idx,:self.nb_points] 

        
        if self.data_aug:
            R = generate_rotation_matrix(theta1*np.pi/2, theta2*np.pi/2, theta3*np.pi/2)
            centering_shift = (self.patch_size-1)/2
            voxels_barycenters = (R@(voxels_barycenters-centering_shift).T).T + centering_shift
            tmp_samplings = np.reshape(((R@np.reshape(tmp_samplings,(-1,3)).T).T), tmp_samplings.shape)


        voxels_samplings = ((tmp_samplings + voxels_barycenters[:,None,:]\
                             - voxels_coordinates[:,None,:]*S)/self.r_mean).astype(FLOAT_TYPE)
    


        return {"image" : torch.tensor(img).unsqueeze(0), "proba" : torch.tensor(proba),
                "voxels_samplings" : torch.tensor(voxels_samplings), "voxels_proba" : torch.tensor(voxels_proba),
                "cell_voxels": cell_voxels}
    


def custom_collate(batch):
    return {
        "images" : torch.stack([item["image"] for item in batch]),
        "proba" : torch.stack([item["proba"] for item in batch]),
        "voxels_samplings" : torch.cat([item["voxels_samplings"] for item in batch]),
        "voxels_proba" : torch.cat([item["voxels_proba"] for item in batch]),
        "cell_voxels" : [item["cell_voxels"] for item in batch]
    }
