import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from torch.nn.functional import conv3d
import torch
from random import randint
from scipy.signal.windows import gaussian


### Creation of the convolution kernel used for fast-marching computation

sampling_kernel = torch.zeros((18, 1, 3, 3, 3), dtype=torch.float32)
idx_no = [0,2,6,8,13,18,20,24,26]
count = 0
for p in range(27):
    if p not in idx_no:
        i, tmp = p%3, p//3
        j, k = tmp%3, tmp//3
        sampling_kernel[count, 0, i,j,k] = 1
        count += 1

def gkern(std = (3,3,3), th = 0.05):
    """Returns a 3D Gaussian kernel array."""
    std_v = np.array(std)
    std_x, std_y, std_z = std
    kerlen = (6*std_v).astype(int)
    kerlen += 1*(kerlen%2==0)
    lx, ly, lz = kerlen
    ker_x = gaussian(lx, std=std_x)[:, None, None]
    ker_y = gaussian(ly, std=std_y)[None, :, None]
    ker_z = gaussian(lz, std=std_z)[None, None, :]

    ker = ker_x*ker_y*ker_z

    return (ker>th)*ker


def farthest_point_sampling(contour_mask, nb_points = 51, anisotropy_ratio = None, device : str = "cpu" ):
    if anisotropy_ratio == None:
        anisotropy_ratio = torch.ones(())

    device_kernel = sampling_kernel.to(device)

    nx,ny,nz = contour_mask.shape
    N = contour_mask.sum()
    i, j, k = 0, 0, 0

    # finding starting point
    while i<nx and contour_mask[i,j,k]==0:
        while j<ny and contour_mask[i,j,k]==0:
            while k<nz and contour_mask[i,j,k]==0:
                k+=1
            if k>=nz:
                j+=1
                k = 0
        if j>=ny:
            i+=1
            j=0

    sampling_list = [[i,j,k]]

    contour_tensor = torch.tensor(contour_mask, dtype=torch.float32, device=device).unsqueeze(0)

    # Adding points to the sampling list
    while len(sampling_list) < nb_points:

        old_fast_marching = torch.zeros_like(contour_tensor)
        new_fast_marching = old_fast_marching.clone()
        for i,j,k in sampling_list:
            new_fast_marching[...,i,j,k] = N

        # Computing fast marching algorithm
        while (new_fast_marching - old_fast_marching).sum()!=0:
            old_fast_marching = new_fast_marching.clone()
            new_fast_marching = (conv3d(new_fast_marching, device_kernel, padding="same").max(dim=0)[0]-1)*contour_tensor
            new_fast_marching = torch.max(old_fast_marching, new_fast_marching)

        # Choosing one of the farthest point to the sampling subset
        tmp = (torch.ones_like(contour_tensor)-contour_tensor)*N + new_fast_marching
        mini_list = (tmp==tmp.min()).nonzero() # prendre un indice random plutot que le premier
        _,i,j,k = mini_list[randint(0,len(mini_list)-1)]

        # Adding this point to the set
        sampling_list.append(torch.tensor([i,j,k]).cpu().numpy())        

    return sampling_list


distance_kernel = torch.zeros((6, 1, 3, 3, 3), dtype=torch.float32)
for idx, value in enumerate([4,10,12,14,16,22]):
    i, tmp = value%3, value//3
    j, k = tmp%3, tmp//3
    distance_kernel[idx, 0, i,j,k] = 1


def distance_to_center(mask):

    eroded = mask
    proba = 0
    
    while eroded.sum()>0:
        proba += eroded
        eroded = binary_erosion(eroded)

    return proba/proba.max()


def mask_to_contour(mask, mode  : str = "erosion"):
    if mode == "erosion":
        eroded = binary_erosion(mask)
        return mask - eroded
    if mode == "dilation":
        dilat = binary_dilation(mask)
        return dilat - mask

def compute_barycenter(mask, mesh):
    mx, my, mz = mesh
    N = mask.sum()

    # cell barycenter coordinates
    x = round((mask*mx).sum()/N) 
    y = round((mask*my).sum()/N)
    z = round((mask*mz).sum()/N)

    return (x,y,z)

def compute_radius(contour, mesh, barycenter):
    mx, my, mz = mesh
    N = contour.sum()
    x,y,z = barycenter

    # radius of the cell
    r = np.sqrt(((mx-x)**2+(my-y)**2+(mz-z)**2)*contour).sum()/N

    return r

def compute_sigma(mask, mesh, barycenter):

    mx, my, mz = mesh
    x, y, z = barycenter
    N = mask.sum()

    rx = (np.abs(mx-x)*mask).sum()/N 
    ry = (np.abs(my-y)*mask).sum()/N
    rz = (np.abs(mz-z)*mask).sum()/N
    
    return rx, ry, rz

def bound_box(mask, mesh):
    mx, my, mz = mesh
    
    x_unique = np.unique(mask*(mx+1))
    x_min, x_max = x_unique[1]-1, x_unique[-1]-1

    y_unique = np.unique(mask*(my+1))
    y_min, y_max = y_unique[1]-1, y_unique[-1]-1

    z_unique = np.unique(mask*(mz+1))
    z_min, z_max = z_unique[1]-1, z_unique[-1]-1

    return x_min, x_max, y_min, y_max, z_min, z_max


    