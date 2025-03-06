import torch
from torch.nn.functional import conv3d

### Creation of the convolution kernel used for fast-marching computation
conv_kernel = torch.zeros((26, 1, 3, 3, 3))
for p in range(26):
    q = p + 1*(p>12)
    i, tmp = q%3, q//3
    j, k = tmp%3, tmp//3
    conv_kernel[p, 0, i,j,k] = -1
    conv_kernel[p, 0, 1,1,1] = 1


def find_centers(spots_map, threshold):
    comparison_map = conv3d(spots_map.float().unsqueeze(0), conv_kernel.to(spots_map.device), padding=1)
    min_map = comparison_map.min(dim=0)[0]
    #max_map = comparison_map.max(dim=0)[0]
    return ((min_map>=0)*(spots_map>threshold)).nonzero()

    
def inside_contours(contours, img_dim):
    """
        - contours : (Batch, nb_facets, nb_points(3), dim_points(3))
    """

    nx, ny, nz = img_dim 
    vx, vy, vz = torch.arange(0, nx), torch.arange(0, ny), torch.arange(0, nz)
    mx, my, mz = torch.meshgrid(vx,vy,vz, indexing="ij")
    voxels = torch.stack((mx,my,mz)).permute(1,2,3,0).to(contours.device)

    v1, v2, v3 = contours[:,:,0], contours[:,:,1], contours[:,:,2] # shape (batch, nb_facets, 3)
    p1, p2 = v1-v2, v3-v2                                          # same
    n = torch.cross(p1,p2,dim=-1)                                         # same
    num = (v2*n).sum(dim=-1)

    B1 = torch.concat((p1[...,None],p2[...,None],n[...,None]), dim = -1) # attention a bien mettre les vecteurs de la base en colonne
    B2 = torch.linalg.inv(B1)[...,:2,:]

    in_map_list = list()

    for i,contour in enumerate(contours):

        in_cell = torch.zeros(size = img_dim).to(contours.device)

        for j, facet in enumerate(contour):

            tmp = n[None,None,None,i,j,:]*voxels
            t = (num[None,None,None,i,j]-tmp[...,2])/tmp[...,:2].sum(dim=-1)

            cdt1 = (t<=1)

            u = torch.concat((voxels[...,:2]*t[...,None], voxels[...,2,None]), dim = -1)

            mat_psg = B2[i,j]

            proj = (mat_psg[None,None,None,...]*(u - v2[None,None,None,i,j])[...,None,:]).sum(dim=-1)

            cdt2 = proj[...,0]>0
            cdt3 = proj[...,1]>0
            cdt4 = proj.sum(dim=-1)<1

            cross_facet = cdt1*cdt2*cdt3*cdt4  

            #print(cross_facet.shape)

            in_cell += cross_facet          
        
        in_map_list.append(in_cell.remainder(2) == 1)
        
    
    return torch.stack(in_map_list)

def inside_contour_slow(contour, patch_dim):

    nx, ny, nz = patch_dim 
    vx, vy, vz = torch.arange(0, nx), torch.arange(0, ny), torch.arange(0, nz)
    mx, my, mz = torch.meshgrid(vx,vy,vz, indexing="ij")
    voxels = torch.stack((mx,my,mz)).permute(1,2,3,0).to(contour.device)

    v1, v2, v3 = contour[:,0], contour[:,1], contour[:,2] # shape (nb_facets, 3)
    p1, p2 = v1-v2, v3-v2                                          # same
    n = torch.cross(p1,p2, dim=-1)                                         # same
    num = (v2*n).sum(dim=-1)

    B1 = torch.concat((p1[...,None],p2[...,None],n[...,None]), dim = -1) # attention a bien mettre les vecteurs de la base en colonne
    B2 = torch.linalg.inv(B1)[...,:2,:]

    in_cell = torch.zeros(size = patch_dim).to(contour.device)

    for j in range(len(contour)):
        tmp = n[None,None,None,j,:]*voxels
        t = (num[None,None,None,j]-tmp[...,2])/tmp[...,:2].sum(dim=-1)

        cdt1 = (t<=1)

        u = torch.concat((voxels[...,:2]*t[...,None], voxels[...,2,None]), dim = -1)

        mat_psg = B2[j]

        proj = (mat_psg[None,None,None,...]*(u - v2[None,None,None,j])[...,None,:]).sum(dim=-1)

        cdt2 = proj[...,0]>=0
        cdt3 = proj[...,1]>=0
        cdt4 = proj.sum(dim=-1)<=1

        cross_facet = cdt1*cdt2*cdt3*cdt4

        in_cell += cross_facet
    
    return (in_cell.remainder(2) == 1)

def inside_contour(contour, patch_dim):
    # faster version that work on a patch of the size if the contour
    nb_facets, _, _ = contour.shape

    nx, ny, nz = patch_dim 
    vx, vy, vz = torch.arange(0, nx), torch.arange(0, ny), torch.arange(0, nz)
    mx, my, mz = torch.meshgrid(vx,vy,vz, indexing="ij")
    voxels = torch.stack((mx,my,mz)).permute(1,2,3,0)[...,None,:].to(contour.device) # shape (mx,my,mz,nb_facets,3)

    v1, v2, v3 = contour[:,0], contour[:,1], contour[:,2] # shape (nb_facets, 3)
    p1, p2 = v1-v2, v3-v2                                          # same
    n = torch.cross(p1,p2, dim=-1)                                         # same
    num = (v2*n).sum(dim=-1)

    B1 = torch.concat((p1[...,None],p2[...,None],n[...,None]), dim = -1) # attention a bien mettre les vecteurs de la base en colonne
    B2 = torch.linalg.inv(B1)[...,:2,:]

    tmp = n[None,None,None,...]*voxels # a voir si ce vecteur n'est pas trop grand
    t = (num[None,None,None,...]-tmp[...,2])/tmp[...,:2].sum(dim=-1)

    u = torch.concat((voxels[...,:2]*t[...,None], voxels[...,2,None].expand(-1,-1,-1,nb_facets,-1)), dim = -1)

    cdt1 = (t<=1) #*(t>=0)

    proj = (B2[None,None,None,...]*(u - v2[None,None,None,...])[...,None,:]).sum(dim=-1)

    cdt2 = proj[...,0]>0
    cdt3 = proj[...,1]>0
    cdt4 = proj.sum(dim=-1)<1

    return (cdt1*cdt2*cdt3*cdt4).sum(dim =-1).remainder(2) == 1


def get_mini_maxi(points):
    mini = points.min(dim=0)[0].floor().to(int)
    maxi = points.max(dim=0)[0].ceil().to(int)

    return mini, maxi









        





