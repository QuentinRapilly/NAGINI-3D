import torch
import numpy as np
from datetime import datetime
from os.path import join, isdir
from os import mkdir
from omegaconf import OmegaConf
from csbdeep.utils import normalize
from scipy.optimize import minimize_scalar
from tqdm import tqdm

from .unet.u3D_3D import Unet3D_3D
from .loss.snake_regularized import RegularizedSnakeLoss
from .tools.snake_sampler import SnakeSmoothSampler
from .tools.display_val import PointCloudDisplay
from .tools.coordinates import spherical_to_xyz
from .tools.inference_tools import find_centers, inside_contour, inside_contour_slow, get_mini_maxi
from .tools.cost_matrix import compute_jaccard
from .tools.refinement import image_to_refinement_grad, evaluate_grad


class Nagini3D():
    def __init__(self, unet_cfg : dict, P : int = 101, M1 : int  = 4, M2 : int = 2,\
                 device : str = "cpu", save_path : str = ".", use_scale = True) -> None:
        self.P = P
        self.M1 = M1
        self.M2 = M2
        self.nb_free_parameters = M1*(M2-1) + 6

        self.device = device

        self.use_scale = use_scale

        self.sampler = SnakeSmoothSampler(P,M1,M2,device)
        self.points_displayer = PointCloudDisplay(M1=M1, M2=M2)

        self.model = Unet3D_3D(num_classes = 3*self.nb_free_parameters + 1, **unet_cfg)
        self.model = self.model.to(device)

        self.save_path = save_path

        
    def init_optimizer(self, optimizer_cfg: dict):
        self.optimizer = {"adam" : torch.optim.AdamW, "sgd" : torch.optim.SGD}[optimizer_cfg["name"]](self.model.parameters(), **optimizer_cfg["parameters"])

    def init_loss(self, loss_lambdas, reg_ratio):
        self.lambdas = loss_lambdas

        self.criterion = {
            "spots" : torch.nn.BCEWithLogitsLoss(),
            "snakes" : RegularizedSnakeLoss(self.device, reg_ratio=reg_ratio)
        }

    def update_snake_loss(self, epoch):
         return self.criterion["snakes"].update_reg_factor(epoch)

    def set_name_and_save_dir(self, exp_name):
        self.exp_name = exp_name
        now = datetime.now()
        self.run_name = f"{exp_name}_{str(now).replace(' ','_').split('.')[0]}"

        self.save_dir = join(self.save_path, self.run_name)

    def reload_save_dir(self, run_name):
        self.run_name = run_name
        self.save_dir = join(self.save_path, run_name)

    def get_run_name(self):
        return self.run_name
    
    def get_save_dir(self):
        return self.save_dir
    
    def save_model(self, model_name : str, epoch, wandb_id, val_loss):
        if not isdir(self.save_dir):
            mkdir(self.save_dir)
        checkpoint = { 
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'wandb_id': wandb_id,
            'val_loss': val_loss}
        torch.save(checkpoint, join(self.save_dir, f"{model_name}_checkpoint.pkl"))

        torch.save(self.model.state_dict(), join(self.save_dir, f"{model_name}.pkl"))
    def save_config(self, cfg_dict : dict):
        if not isdir(self.save_dir):
            mkdir(self.save_dir)
        OmegaConf.save(config=cfg_dict, f=join(self.save_dir, "config.yaml"))

    def save_th(self, th_dict : dict):
        if not isdir(self.save_dir):
            mkdir(self.save_dir)
        OmegaConf.save(config=th_dict, f=join(self.save_dir, "thresholds.yaml"))


    def load_weights(self, weights_file):
        """
        Load in the model the weights stored during a training

        Input:
        -weights_file: path to the weights file
        """
        self.model.load_state_dict(torch.load(weights_file, map_location=self.device, weights_only=True))
    
    def load_checkpoint(self, checkpoint_path, optimizer_cfg):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model"])

        optimizer_constructor = {"adam" : torch.optim.AdamW, "sgd" : torch.optim.SGD}[optimizer_cfg["name"]]
        self.optimizer = optimizer_constructor(self.model.parameters(), **optimizer_cfg["parameters"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        return checkpoint["epoch"], checkpoint["wandb_id"], checkpoint["val_loss"]


    def normalize_img(self, img):
        return torch.tensor(normalize(img, pmin=1, pmax=99.8, axis=(0,1,2), clip=True), dtype=torch.float32, device=self.device)
    

    def epoch(self, data_loader : torch.utils.data.DataLoader):
        """
        Run a training epoch and return a dictionnary storing the different losses.
        """
        
        infos = {"spots" : 0, "snakes" : 0, "regularization" : 0, "loss" : 0}

        nb_batches = len(data_loader) # Nb of batches

        for idx_batch, batch in enumerate(data_loader):
            print(f"Progression {round(idx_batch/nb_batches*100)}%\r", end="")

            self.optimizer.zero_grad()

            imgs = batch["images"].to(device=self.device)
            proba = batch["proba"].to(device=self.device)
            voxels_proba = batch["voxels_proba"].to(device=self.device)
            cell_voxels = batch["cell_voxels"]

            samplings = batch["voxels_samplings"].to(device=self.device)

            with torch.autocast(device_type="cuda"):

                features, _ = self.model(imgs)

                pred_spots_maps = features[:,0]
                free_parameters = features[:,1:]

                cartesian_list = list()
                scale_list = list()

                for idx_sample, sample in enumerate(free_parameters):
                    x_cell, y_cell, z_cell = cell_voxels[idx_sample]
                    free_param_at_centers = sample[...,x_cell,y_cell, z_cell]
                    scale_factor = free_param_at_centers[0]  # experiment on scale factor 
                    free_param_at_centers = free_param_at_centers[1:] # experiment on scale factor


                    x,y,z = spherical_to_xyz(free_param_at_centers)
                    cp = torch.stack((x,y,z)).permute(2,1,0)
                    
                    cartesian_list.append(cp)
                    scale_list.append(scale_factor)

                free_parameters_cartesian = torch.cat(cartesian_list)
                all_factors = torch.cat(scale_list) if self.use_scale else torch.tensor([1], device=self.device)

                pred_samplings = self.sampler.sample_snakes(free_parameters_cartesian)*all_factors[...,None,None]
                ds_du, ds_dv = self.sampler.get_derivatives(free_parameters_cartesian*all_factors[...,None,None])

                spots_loss = self.criterion["spots"](pred_spots_maps, proba)
                loss = spots_loss

                nb_cell_voxel = len(pred_samplings)
                if nb_cell_voxel>0: # do not compute the snake and reg loss if there is no cell voxel in the batch (to avoid NaN)
                    snakes_loss = self.criterion["snakes"](pred_samplings, samplings, voxels_proba, ds_du, ds_dv)
                    scale_loss = torch.nn.functional.mse_loss(all_factors, samplings.norm(dim=-1).mean(dim=-1)) if self.use_scale else 0
                
                    loss = self.lambdas["spots"]*loss + self.lambdas["snakes"]*snakes_loss + self.lambdas["scale"]*scale_loss

            loss.backward()
            self.optimizer.step()

            if nb_cell_voxel>0:
                infos["snakes"] += snakes_loss.item()/nb_batches
                infos["scale"] += (scale_loss.item()/nb_batches) if self.use_scale else 0
            infos["spots"] += spots_loss.item()/nb_batches
            infos["loss"] += loss.item()/nb_batches
        return infos
    
    def val(self, data_loader : torch.utils.data.DataLoader, nb_cells_to_plot : int = 4):
        """
        Run a validation step and return a dictionnary storing the different losses.

        Input:
        - data_loader: validation data loader,
        - nb_cells_to_plot: if WandB is used, this parameter correspond to the number of 3D point clouds to plot to visualize the accuracy of the predicted surfaces.
        """
        infos = {"spots" : 0, "snakes" : 0, "regularization" : 0, "loss" : 0}
        nb_batches = len(data_loader) # Nb of batches

        with torch.no_grad():
            for idx_batch, batch in enumerate(data_loader):
                print(f"Progression {round(idx_batch/nb_batches*100)}%\r", end="")

                imgs = batch["images"].to(device=self.device)
                proba = batch["proba"].to(device=self.device)
                voxels_proba = batch["voxels_proba"].to(device=self.device)
                cell_voxels = batch["cell_voxels"]

                samplings = batch["voxels_samplings"].to(device=self.device)

                features, _ = self.model(imgs)

                pred_spots_maps = features[:,0]
                free_parameters = features[:,1:]

                cartesian_list = list()
                scale_list = list()

                for idx_sample, sample in enumerate(free_parameters):
                    x_cell, y_cell, z_cell = cell_voxels[idx_sample]
                    free_param_at_centers = sample[...,x_cell,y_cell, z_cell]
                    scale_factor = free_param_at_centers[0]  # experience on scale factor
                    free_param_at_centers = free_param_at_centers[1:]  # experience on scale factor

                    
                    x,y,z = spherical_to_xyz(free_param_at_centers)
                    cp = torch.stack((x,y,z)).permute(2,1,0)
                    
                    cartesian_list.append(cp)
                    scale_list.append(scale_factor)  # experience on scale factor

                free_parameters_cartesian = torch.cat(cartesian_list)
                all_factors = torch.cat(scale_list) if self.use_scale else torch.tensor([1], device=self.device) # experience on scale factor

                pred_samplings = self.sampler.sample_snakes(free_parameters_cartesian)*all_factors[...,None,None]
                ds_du, ds_dv = self.sampler.get_derivatives(free_parameters_cartesian*all_factors[...,None,None])

                bg_mask = (proba==0)[:,None,...]

                spots_loss = self.criterion["spots"](pred_spots_maps, proba)
                loss = spots_loss
                
                nb_cell_voxel = len(pred_samplings)
                if nb_cell_voxel>0: # do not compute the snake and reg loss if there is no cell voxel in the batch (to avoid NaN)

                    snakes_loss = self.criterion["snakes"](pred_samplings, samplings, voxels_proba, ds_du, ds_dv)
                    scale_loss = torch.nn.functional.mse_loss(all_factors, samplings.norm(dim=-1).mean(dim=-1)) if self.use_scale else 0

                    loss = self.lambdas["spots"]*loss + self.lambdas["snakes"]*snakes_loss + self.lambdas["scale"]*scale_loss

                if nb_cell_voxel>0:
                    infos["snakes"] += snakes_loss.item()/nb_batches
                    infos["scale"] += (scale_loss.item()/nb_batches) if self.use_scale else 0
                infos["spots"] += spots_loss.item()/nb_batches
                infos["loss"] += loss.item()/nb_batches

            
            idx_pseudo_centers = (voxels_proba == 1)

            cloud_list = self.points_displayer.display_val(nb_cells=nb_cells_to_plot, GT=samplings[idx_pseudo_centers].cpu(),\
                                            pred=pred_samplings[idx_pseudo_centers].cpu(), cp=free_parameters_cartesian[idx_pseudo_centers].cpu())

            return infos, cloud_list
        
    

    def apply_network(self, img, proba_th, GT_centers = None, predict_centers = True, nb_tiles = (1,1,1), overlap_ratio = 0.1):

        """
        Apply the UNet to get the score map and the surface parameters and return the surface parameters at the score map local maxima.
        
        Input:
        - img: np.array(width,height,depth), 3D image to segment
        - proba_th: float, threshold used for object detection (computed automatically on validation set)
        - GT_centers: if provided, will also return the surface parameters at the provided centers
        - predict_centers: bool, if True detect the centers (maxima of the score map)
        - nb_tiles: (int, int,int), number of splits on each dimension to make tiles with the image if it is too big to process in a single time
        - overlap_ratio: float in [0,1], ratio of the tile size to overlap with the previous/next tile 
        
        Output:
        - dict{"proba", "pred"{"centers", "surface_parameters"}, "GT"{"centers", "surface_parameters"}}
        """

        img = self.normalize_img(img)

        Nx, Ny, Nz = img.shape
        nx, ny, nz = nb_tiles

        lx, ly, lz = (Nx//nx) + (Nx%nx > 0), (Ny//ny) + (Ny%ny > 0), (Nz//nz) + (Nz%nz > 0)
        ox, oy, oz = int(lx*overlap_ratio), int(ly*overlap_ratio), int(lz*overlap_ratio)

        centers_list = list()
        parameters_list = list()

        if GT_centers != None:
            GT_param_list = list()
            GT_index_order = list()
            X_GT, Y_GT, Z_GT = GT_centers.T

        proba_pred = torch.zeros_like(img, dtype=float, device=self.device)
        proba_votes = torch.zeros_like(img, dtype=int, device=self.device)

        result = {}

        with torch.no_grad():

            for ix in range(nx):
                X_min, X_max = lx*ix, lx*(ix + 1)
                X_left, X_right = max(0, X_min-ox), min(Nx, X_max+ox)

                for iy in range(ny):
                    Y_min, Y_max = ly*iy, ly*(iy + 1)
                    Y_left, Y_right = max(0, Y_min-oy), min(Ny, Y_max+oy)

                    for iz in range(nz):
                        Z_min, Z_max = lz*iz, lz*(iz + 1)
                        Z_left, Z_right = max(0, Z_min-oz), min(Nz, Z_max+oz)

                        if True :#verbose : 
                            print(f"Infering tile : X : {[X_left,X_right]}, Y : {[Y_left, Y_right]}, Z : {[Z_left, Z_right]}")

                        features, _ = self.model(img[None,None,X_left:X_right,Y_left:Y_right,Z_left:Z_right])

                        pred_proba = torch.sigmoid(features[0,0])
                        free_parameters_spherical = features[0,2:]
                        scale_factor = features[0,1]

                        proba_pred[...,X_left:X_right,Y_left:Y_right,Z_left:Z_right] += pred_proba
                        proba_votes[...,X_left:X_right,Y_left:Y_right,Z_left:Z_right] += 1

                        # Find predicted centers and build the corresponding surfaces
                        if predict_centers:
                            centers = find_centers(pred_proba, proba_th)
                            X,Y,Z = centers.T
                            in_patch = (X>=X_min-X_left)*(X<=X_min-X_left+lx)*\
                                (Y>=Y_min-Y_left)*(Y<=Y_min-Y_left+ly)*\
                                    (Z>=Z_min-Z_left)*(Z<=Z_min-Z_left+lz)
                            X, Y, Z = X[in_patch], Y[in_patch], Z[in_patch]
                            fp_spherical = free_parameters_spherical[..., X, Y, Z]
                            crt_scales = scale_factor[...,X,Y,Z] if self.use_scale else torch.tensor([1], device=self.device)

                            x, y, z = spherical_to_xyz(fp_spherical)

                            fp_cartesian = torch.stack((x,y,z)).permute(2,1,0)*crt_scales[...,None,None]

                            centers_list.append(centers[in_patch] + torch.tensor([[X_left, Y_left, Z_left]]).to(self.device))
                            parameters_list.append(fp_cartesian.to(self.device))

                        # If some GT centers where given, build the surfaces at those positions
                        if GT_centers != None:
                            GT_in_patch = (X_GT>=X_min)*(X_GT<=X_max)*\
                                (Y_GT>=Y_min)*(Y_GT<=Y_max)*\
                                    (Z_GT>=Z_min)*(Z_GT<=Z_max)
                            GT_index_order.append(torch.arange(0,len(GT_centers),step=1)[GT_in_patch]) # maybe require a +1 but should not with the new version
                        
                            X_GT_crt, Y_GT_crt, Z_GT_crt = GT_centers[GT_in_patch].T
                            X_GT_shifted, Y_GT_shifted, Z_GT_shifted = X_GT_crt - X_left, Y_GT_crt - Y_left, Z_GT_crt - Z_left

                            GT_spherical = free_parameters_spherical[..., X_GT_shifted, Y_GT_shifted, Z_GT_shifted]
                            x_GT, y_GT, z_GT = spherical_to_xyz(GT_spherical)
                            GT_cartesian = torch.stack((x_GT, y_GT, z_GT)).permute(2,1,0)

                            GT_param_list.append(GT_cartesian.to(self.device))


        # Processing probability prediction
        proba = (proba_pred/proba_votes)

        result["proba"] = proba

        # Processing surface at predicted centers
        if predict_centers:
            all_centers = torch.cat(centers_list, dim=0)
            all_parameters = torch.cat(parameters_list, dim=0)

            prob_centers = proba[all_centers[:,0],all_centers[:,1],all_centers[:,2]]
            _, idx_sort = torch.sort(prob_centers, descending=True)

            all_centers = all_centers[idx_sort]
            all_parameters = all_parameters[idx_sort]

            result["pred"] = {"centers" : all_centers, "surface_parameters" : all_parameters}

        if GT_centers != None:
            all_GT_param = torch.cat(GT_param_list, dim=0)
            all_GT_idx = torch.cat(GT_index_order, dim=0)
            _, order = torch.sort(all_GT_idx)
            all_GT_param = all_GT_param[order]

            result["GT"] = {"centers" : GT_centers, "surface_parameters" : all_GT_param}

        return result
    
    def create_mask(self, img_shape, centers, samplings, facets, nms_th = None, d_mask = 3):

        """
        Create a mask with the same shape as the input image with labels representing the inside of each predicted surfaces.
        
        Input:
        - img_shape: (int,int,int), shape of the initial input image;
        - centers: torch.tensor(n,3), centers of the predicted surfaces
        - samplings: torch.tensor(n,P,3), points sampled on the predicted surfaces
        - facets: torch.tensor(K,3), facets represented by the indexes of the points which are vertices of the facet,
        - nms_th: float in [0,1], Non-Maximum-Suppression threshold, if two object have an IoU greater only one will be kept
        """

        mask_pred = torch.zeros(img_shape, dtype=torch.int32, device=self.device)
        Nx, Ny, Nz = img_shape
        nb_centers = len(centers)
        crt_idx = 1

        remaining_idx = list()
        removed_idx = list()

        with torch.no_grad():
            for i in range(nb_centers):
                s_i = samplings[i] + centers[i,None,:]
                mini, maxi = get_mini_maxi(s_i)
                contour_i = (s_i - mini[None,:] + d_mask)[facets]
                try :
                    patch_mask = inside_contour(contour_i, (maxi-mini+2*d_mask).tolist())
                except :
                    patch_mask = inside_contour_slow(contour_i, (maxi-mini+2*d_mask).tolist())

                patch = patch_mask[d_mask+max(0,-mini[0]):-(d_mask+max(0,maxi[0]-Nx)),
                                   d_mask+max(0,-mini[1]):-(d_mask+max(0,maxi[1]-Ny)),
                                   d_mask+max(0,-mini[2]):-(d_mask+max(0,maxi[2]-Nz))]
                at_patch = mask_pred[max(0,mini[0]):maxi[0], max(0,mini[1]):maxi[1], max(0,mini[2]):maxi[2]]
                fg = at_patch>0
                bg = at_patch==0
                cell_volume = patch.sum()
                overlap = (patch*fg).sum()
                if overlap/cell_volume<nms_th:
                    mask_pred[max(0,mini[0]):maxi[0], max(0,mini[1]):maxi[1], max(0,mini[2]):maxi[2]] += bg*crt_idx*patch
                    crt_idx += 1
                    remaining_idx.append(i)
                else:
                    removed_idx.append(i)

        remaining_centers = centers[remaining_idx]
        removed_centers = centers[removed_idx]

        samp_to_save = (samplings + centers[:,None,:])[remaining_idx]
        
        return mask_pred, samp_to_save, remaining_centers, removed_centers, remaining_idx
    

    def optimize_snake(self, img, centers, surf_param, gamma = 0.01, nb_steps = None, grad_fn = image_to_refinement_grad):

        """
        Apply the snake refinement step on the image using the previously computed surfaces.

        Input:
        - img: np.array(width,height,depth), 3D image used to apply the refinement process
        - centers: np.array(n,3), centers of all the predicted surfaces
        - surf_param: np.array(n,P,3), surface parameters/control points of all the predicted surfaces
        - gamma: float, step of the gradient optimization of the surface parameters
        - n_steps: int(optional), number of steps applied to optimimze the surface parameters (if not provided, computed using gamma)
        - grad_fn: np.array(width, heigh, depth) -> [np.array(width, heigh, depth),np.array(width, heigh, depth),np.array(width, heigh, depth)]
                funtion which gives the displacement (dX, dY, dZ) to apply to the surface in each location of the image

        Output:
        - tuple(points, facets, values), parameters of the optimized surfaces, points are the position of sampled points on the surface, 
        """

        if nb_steps == None: nb_steps = int(5/gamma)

        norm_img = normalize(img, pmin=1, pmax=99.8, axis=(0,1,2), clip=True)

        grad_X, grad_Y, grad_Z = grad_fn(norm_img)

        surf_param_cp = surf_param


        ds_dfi = self.sampler.fi_weights

        for _ in range(nb_steps):
            samples = (self.sampler.sample_snakes(surf_param_cp) + centers[:,None,:]).cpu()

            dX, dY, dZ = evaluate_grad(samples, grad_X, grad_Y, grad_Z)

            grad = torch.tensor(np.stack((dX,dY,dZ)).transpose((1,2,0))).to(self.device)

            Dfi = (ds_dfi*grad[...,None,:]).sum(dim=(1))

            surf_param_cp = surf_param_cp - gamma*Dfi

        generated_surf = self.sampler.draw_surface(surf_param_cp , points_per_dim=(20,10))
        return surf_param_cp, generated_surf
    

    def inference(self, img, proba_th, r_mean, nb_tiles, overlap_ratio = 0.1, nms_th = 0.4, 
                  optim_snake = True, grad_fn = image_to_refinement_grad, d_mask = 3, anisotropy = [1,1,1]):
        """
        Apply the model to a new image.
        
        Input:
        - img: np.array(width,height,depth), 3D image to segment
        - proba_th: float, threshold used for object detection (computed automatically on validation set)
        - r_mean: float, mean radius of the objects used to train the model (stored in the config file during training)
        - nb_tiles: (int, int,int), number of splits on each dimension to make tiles with the image if it is too big to process in a single time
        - overlap_ratio: float in [0,1], ratio of the tile size to overlap with the previous/next tile 
        - nms_th: float in [0,1], Non-Maximum-Suppression threshold, if two object have an IoU greater only one will be kept
        - optim_snake: bool, weither to process snake refinement after network prediction
        - grad_fn: np.array(width, heigh, depth) -> [np.array(width, heigh, depth),np.array(width, heigh, depth),np.array(width, heigh, depth)]
                funtion which gives the displacement (dX, dY, dZ) to apply to the surface in each location of the image for the refinement 

        Output:
        - mask: np.array(width,height,depth), 3D mask associating a label to each voxel
        - proba: np.array(width,height,depth), 3D map associating a score to be the center of an object to each voxel
        - points: dict{"points", "facets", "values", "centers", "params"}, a dict containing the points, facets and values required to reconstruct the surfaces using napari
        and the surface centers and shape parameters to provide to the shape analysis features.
        """

        with torch.no_grad():
            result = self.apply_network(img, proba_th=proba_th, nb_tiles=nb_tiles, overlap_ratio=overlap_ratio)
        
        anisotropy_vec = np.array(anisotropy)
        anisotropy_ten = torch.tensor(anisotropy_vec, device=self.device)[None,None,...]

        centers, surf_parameters = result["pred"]["centers"], result["pred"]["surface_parameters"]

        if optim_snake:
            params, new_surfaces = self.optimize_snake(img, centers, r_mean*surf_parameters/anisotropy_ten, grad_fn=grad_fn)
            samplings, facets, values = new_surfaces
        else:
            params = r_mean*surf_parameters/anisotropy_ten
            samplings, facets, values = self.sampler.draw_surface(free_parameters=params, points_per_dim=(20,10))
        
        mask, samplings, remain_centers, _, remain_idx = self.create_mask(img.shape, centers=centers, samplings=samplings,\
                                           nms_th=nms_th, facets=facets, d_mask=d_mask)

        return mask.cpu().numpy(), result["proba"].cpu().numpy(),\
            {"points" : samplings.cpu().numpy(), "facets" : facets.cpu().numpy(), "values" : values.cpu().numpy(),
             "params": params[remain_idx].cpu().numpy(), "centers" :remain_centers.cpu().numpy()}
    

    def inference_classic_n_refined(self, img, proba_th, r_mean, nb_tiles, overlap_ratio = 0.1, nms_th = 0.4, 
                  grad_fn = image_to_refinement_grad, d_mask = 3, anisotropy = [1,1,1]):
        
        with torch.no_grad():
            result = self.apply_network(img, proba_th=proba_th, nb_tiles=nb_tiles, overlap_ratio=overlap_ratio)
        
        anisotropy_vec = np.array(anisotropy)
        anisotropy_ten = torch.tensor(anisotropy_vec, device=self.device)[None,None,...]

        centers, surf_parameters = result["pred"]["centers"], result["pred"]["surface_parameters"]

        params_before = r_mean*surf_parameters/anisotropy_ten
        samp_before, facets_before, values_before = self.sampler.draw_surface(free_parameters=params_before, points_per_dim=(20,10))

        before_mask, final_classic_samp, remain_centers_classic, _, remain_idx_classic =\
            self.create_mask(img.shape, centers=centers, samplings=samp_before, nms_th=nms_th,
                             facets=facets_before, d_mask=d_mask)
        
        params_after, new_surfaces = self.optimize_snake(img, centers, params_before, grad_fn=grad_fn)
        samp_after, facets_after, values_after = new_surfaces

        after_mask, final_refined_samp, remain_centers_refined, _, remain_idx_refined = \
            self.create_mask(img.shape, centers=centers, samplings=samp_after, nms_th=nms_th,
                             facets=facets_after, d_mask=d_mask)
        
        return before_mask.cpu().numpy(), after_mask.cpu().numpy(), result["proba"].cpu().numpy(),\
            {"points" : final_classic_samp.cpu().numpy(), "facets" : facets_before.cpu().numpy(),
             "values" : values_before.cpu().numpy(), "params": params_before[remain_idx_classic].cpu().numpy(),
             "centers" :remain_centers_classic.cpu().numpy()},\
            {"points" : final_refined_samp.cpu().numpy(), "facets" : facets_after.cpu().numpy(),
             "values" : values_after.cpu().numpy(), "params": params_after[remain_idx_refined].cpu().numpy(),
             "centers" :remain_centers_refined.cpu().numpy()}

    

    def predict_on_points(self, img, choosen_points, r_mean, nb_tiles, overlap_ratio = 0.1, d_mask = 3):

        result = self.apply_network(img, proba_th=None, r_mean=r_mean, GT_centers=choosen_points, predict_centers=False,\
                                    nb_tiles=nb_tiles, overlap_ratio=overlap_ratio)
        
        mask, _, _, _ = self.create_mask(img.shape, centers=result["GT"]["centers"], samplings=result["GT"]["samplings"],\
                                           nms_th=None, facets=result["tools"]["facets"], d_mask=d_mask)
        
        return mask
    

    def shapes_curvature(self, params):

        tensor_params = torch.tensor(params, device=self.device)
        curvature = self.sampler.get_curvature(tensor_params)

        mean_curv = curvature.mean(dim=-1)
        std_curv = curvature.std(dim=-1)
        max_curv, _  = curvature.max(dim=-1)
        min_curv, _ = curvature.min(dim=-1)

        #draw_curv = self.sampler.draw_curvature(tensor_params, points_per_dim=(30,15))

        return {"mean_curv" : mean_curv.cpu().numpy(), "std_curv": std_curv.cpu().numpy(), "max_curv": max_curv.cpu().numpy(),
                "min_curv": min_curv.cpu().numpy()}
    

        
    def optimize_thresholds(self, optim_set, r_mean, nms_th = [0.3,0.5,0.7], nb_tiles = (1,1,1),
                            iou_bins = [0.0,0.2,0.4,0.6,0.8,1.0], maxiter = 20):
        """
        Compute the optimal probability and NSM thresholds on a given dataset.

        Input:
        - optim_set: instance of OptimSet(Dataset), dataset on which to apply the optimization process
        - r_mean: float, mean radius of the objects used to train the model (stored in the config file during training)
        - nb_tiles: (int, int,int), number of splits on each dimension to make tiles with the image if it is too big to process in a single time
        - nms_th: list[float], list of possible values tested for NMS threshold

        Output:
        - optim_th: dict{"prob", "nms"}, dictionnary storing the optim proba and NMS threshold computed
        """
        
        proba_max = max([(self.apply_network(optim_set[n][0], proba_th=0.5, predict_centers=False,\
                                             nb_tiles=nb_tiles,))["proba"].max() for n in range(len(optim_set))])

        proba_max = proba_max.item()

        bounds = proba_max*0.1, proba_max*0.95

        best_nms, best_prob, best_f = None, None, 0  

        for nms in nms_th:
            
            with tqdm(total=maxiter, desc=f"NMS threshold = {nms}") as progress:

                def fn(proba_th):

                    TP, N_GT, N_pred = 0, 0, 0
                    
                    N = len(optim_set)

                    for n in range(N):
                        img, mask = optim_set[n]
                        pred_mask, _, _ = self.inference(img, proba_th, r_mean=r_mean, nb_tiles=nb_tiles,\
                                                        nms_th=nms, verbose=False)
                        tp, n_GT, n_pred = compute_jaccard(mask, pred_mask.detach().cpu().numpy(), iou_bins)
                        TP += tp
                        N_GT += n_GT
                        N_pred += n_pred

                    jaccard = TP / (N_GT + N_pred - TP)
                    iou_gaps = np.array(iou_bins[1:])-np.array(iou_bins[:-1])

                    mean_jaccard = (iou_gaps*jaccard/N).sum()

                    progress.update()
                    progress.set_postfix_str(f"prob_th : {proba_th} -> jaccard area : {mean_jaccard}")
                    progress.refresh()

                    return -mean_jaccard # should return the opposite as we use "minimize_scalar" but want to maximize the jaccard

                    
                # opt = minimize_scalar(fn, method='golden', bracket=bracket, tol=tol, bounds=bounds, options={'maxiter': maxiter})
                opt = minimize_scalar(fn, method='bounded', bounds=bounds, options={'maxiter': maxiter})

            opt_prob, opt_f = opt.x, -opt.fun

            if opt_f > best_f:
                best_f, best_nms, best_prob = opt_f, nms, opt_prob


        opt_th = {"prob" : best_prob.item(), "nms" : best_nms}
        self.opt_th = opt_th

        return opt_th

        