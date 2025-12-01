from torch.utils.data import Dataset
from os.path import join, basename, splitext
from glob import glob
import tifffile
import numpy as np
from random import randint
import torch
from typing import Any
from csbdeep.utils import normalize

from data_aug_tools import generate_rotation_matrix, rotate_image

FLOAT_TYPE = np.float32
INT_TYPE = np.uint16

class TrainingSet(Dataset):

    def __init__(self, dataset_dir : str, patch_size : list, nb_points : int = 101, anisotropy_ratio : list = [1,1,1],
                 r_mean = None, data_aug : bool = True, cell_ratio_th = 0.0, **kwargs) -> None:
        super().__init__()

        self.images_dir = join(dataset_dir, "images")
        self.samplings_dir = join(dataset_dir, "samplings")
        self.masks_dir = join(dataset_dir, "masks")

        self.patch_size = patch_size
        self.nb_points = nb_points
        self.data_aug = data_aug

        self.is_isotropic = (anisotropy_ratio == None)
        self.anisotropy_ratio = np.array([[[1,1,1]]]) if self.is_isotropic else np.array(anisotropy_ratio)[None,None,:]

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
        Mx, My, Mz = self.patch_size
        x_max, y_max, z_max = nx-Mx, ny-My, nz-Mz
        x, y, z = randint(0, x_max), randint(0, y_max), randint(0, z_max)
        return img[x:x+Mx, y:y+My, z:z+Mz], (x,y,z)

    def __getitem__(self, index) -> Any:
        img_path = self.imgs_list[index]
        img_name = basename(img_path)
        print(img_name)
        img_name_no_ext = splitext(img_name)[0]


        full_img = tifffile.imread(img_path)
        full_img = normalize(full_img, pmin=1, pmax=99.8, axis=(0,1,2), clip=True)

        full_proba = tifffile.imread(join(self.samplings_dir, img_name))
        full_mask = tifffile.imread(join(self.masks_dir, img_name))

        samplings_arrays = np.load(join(self.samplings_dir, img_name_no_ext+".npz"))

        full_samplings = samplings_arrays["samplings"]
        full_centers = samplings_arrays["centers"]

        Mx, My, Mz = self.patch_size

        mask, (x,y,z) = self.crop_image(full_mask)
        mask = mask.astype(INT_TYPE)

        if self.cell_ratio_th != 0: # avoid to compute the sum when one chooses the ratio equal to 0 (i.e. no verification)
            cell_ratio_in_mask = (mask>0).sum()/(Mx*My*Mz)
            while cell_ratio_in_mask < self.cell_ratio_th:
                mask, (x,y,z) = self.crop_image(full_mask)
                mask = mask.astype(INT_TYPE)
                cell_ratio_in_mask = (mask>0).sum()/(Mx*My*Mz)


        proba = full_proba[x:x+Mx, y:y+My, z:z+Mz].astype(FLOAT_TYPE)
        img = full_img[x:x+Mx, y:y+My, z:z+Mz].astype(FLOAT_TYPE)

        unique_lbls = np.unique(full_mask)[1:]
        lbls = np.arange(unique_lbls[-1]+1, dtype=int)
        lbls[unique_lbls] = np.arange(len(unique_lbls), dtype=int)


        if self.data_aug and self.is_isotropic:
            theta1, theta2, theta3 = randint(0,3), randint(0,3), randint(0,3)
            angles = (theta1, theta2, theta3)

            img = rotate_image(img, angles).copy()
            proba = rotate_image(proba, angles).copy()
            mask = rotate_image(mask, angles).copy()
        

        cell_voxels = mask.nonzero()
        voxels_coordinates = np.stack(cell_voxels).T
        voxels_idx = lbls[mask[cell_voxels]]
        voxels_proba = proba[cell_voxels]
        voxels_barycenters = (full_centers[voxels_idx] - np.array([x,y,z]))

        assert self.nb_points <= full_samplings.shape[1]
        tmp_samplings = full_samplings[voxels_idx,:self.nb_points] 

        
        if self.data_aug and self.is_isotropic:
            R = generate_rotation_matrix(theta1*np.pi/2, theta2*np.pi/2, theta3*np.pi/2)
            centering_shift = (self.patch_size[0]-1)/2
            voxels_barycenters = (R@(voxels_barycenters-centering_shift).T).T + centering_shift
            tmp_samplings = np.reshape(((R@np.reshape(tmp_samplings,(-1,3)).T).T), tmp_samplings.shape)


        voxels_samplings = (((tmp_samplings + voxels_barycenters[:,None,:]\
                             - voxels_coordinates[:,None,:]+0.5)*self.anisotropy_ratio-0.5)/self.r_mean).astype(FLOAT_TYPE)
    


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


if __name__ == "__main__":
    path = "/home/qrapilly/Documents/Code/Data/Elena_CAIC/non-star-conv-set" # "/home/qrapilly/Documents/Code/Data/Simu_2D+t_Charles/SimuFormes3D/filtered_set/tmp_dataset/train" # "/home/qrapilly/Documents/Code/Data/domain_shift/crowns_n_cluster/val" # "/home/qrapilly/Documents/Code/Data/chips_poisson/chips_dense/val"# "/home/qrapilly/Documents/Code/Data/sainte_data/val"
    anisotropy = [2,1,1]
    dataset = TrainingSet(dataset_dir=path, patch_size=[30,80,80], nb_points=101, anisotropy_ratio=anisotropy, cell_ratio_th=0.01)

    item = dataset[1]

    img = item["image"]
    proba = item["proba"]
    samplings = item["voxels_samplings"]
    vox_proba = item["voxels_proba"]
    cell_vox = item["cell_voxels"]

    gamma = (np.array(img.shape)[1:]/np.array(proba.shape))
    coor = (np.stack(cell_vox).T)*gamma

    N = samplings.shape[0]

    points_to_plot = list()

    idx_array = np.argsort(vox_proba)

    random_idx = [k for k in range(N)]
    from random import shuffle
    shuffle(random_idx)

    


    for i in idx_array[-50:]:
        s = samplings[i]
        c = coor[i]
        s_shifted = s*dataset.r_mean + c[None,:]*np.array([anisotropy])
        
        # print(s)
        points_to_plot.append(s_shifted)


    # print(len(cell_vox[0])/(proba==0).sum())

    
    
    import napari
    from skimage.transform import rescale
    img = rescale(img[0].numpy(), anisotropy)
    viewer = napari.view_image(img, ndisplay=2)
    #viewer.add_image(proba.numpy())

    #viewer.add_points(coor, face_color="red", size=1)
    viewer.add_points(np.concatenate(points_to_plot), face_color="red", size=1)
    #viewer.add_points(coor*np.array([anisotropy]),size = 1, face_color="green")
    

    napari.run()