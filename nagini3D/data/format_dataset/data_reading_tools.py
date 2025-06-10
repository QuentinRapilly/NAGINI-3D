import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from torch.nn.functional import conv3d
import torch
from random import randint
from scipy.signal.windows import gaussian


### Creation of the convolution kernel used for fast-marching computation

sampling_kernel = torch.zeros((19, 1, 3, 3, 3), dtype=torch.float32)
idx_no = [0,2,6,8,18,20,24,26]
count = 0
for p in range(27):
    if p not in idx_no:
        k, tmp = p%3, p//3
        j, i = tmp%3, tmp//3
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


def farthest_point_sampling(contour_mask, nb_points = 51, device : str = "cpu", anisotropy = (1,1,1)):
    ax, ay, az = anisotropy
    crt_kern = torch.sqrt(torch.tensor([ax, 0, ax])[:,None,None]**2 + torch.tensor([ay, 0, ay])[None,:,None]**2 + torch.tensor([az, 0, az])[None,None,:]**2)
    idx_ker = torch.ones((27), dtype=bool)
    idx_ker[idx_no] = False
    add_ker = (crt_kern.flatten()[idx_ker]).to(device)[:,None,None,None]

    device_kernel = sampling_kernel.to(device)

    nx,ny,nz = contour_mask.shape
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

    contour_tensor = torch.tensor(contour_mask, dtype=bool, device=device)

    # Adding points to the sampling list
    while len(sampling_list) < nb_points:

        old_fast_marching = torch.zeros_like(contour_tensor, dtype=torch.float32)
        new_fast_marching = old_fast_marching.clone()
        for i,j,k in sampling_list:
            new_fast_marching[i,j,k] = 1 # pseudo_inf

        # Computing fast marching algorithm
        while ((new_fast_marching - old_fast_marching)*contour_tensor).sum()!=0:
            old_fast_marching = new_fast_marching.clone()
            conv_step = (conv3d(new_fast_marching[None,...], device_kernel, padding="same"))
            new_fast_marching = ((conv_step + torch.where(conv_step > 0, add_ker, torch.inf)).min(dim=0)[0])
            new_fast_marching = torch.where((new_fast_marching<torch.inf)*contour_tensor, new_fast_marching, 0)
            #new_fast_marching = new_fast_marching.where(contour_tensor, 0)
            # new_fast_marching = torch.maximum(old_fast_marching, new_fast_marching)
        

        # Choosing one of the farthest point to the sampling subset
        maxi_list = (new_fast_marching==new_fast_marching.max()).nonzero() # prendre un indice random plutot que le premier
        i,j,k = maxi_list[randint(0,len(maxi_list)-1)]

        # Adding this point to the set
        sampling_list.append([i.item(),j.item(),k.item()]) 

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


    