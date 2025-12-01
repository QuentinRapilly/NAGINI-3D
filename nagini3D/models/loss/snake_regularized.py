import torch.nn as nn

from .sliced_wasserstein_loss import SlicedWassersteinLoss


class RegularizedSnakeLoss(nn.Module):

    def __init__(self, device, reg_ratio = 0.01, *args, **kwargs) -> None:
        super().__init__()


        self.device = device
        self.reg_ratio = reg_ratio

        self.snake_loss = SlicedWassersteinLoss(nb_angles=100, W_order=1, p_order=1, device=self.device)

    
    def forward(self, sampling, target, proba, ds_du, ds_dv):

        snake = self.snake_loss(sampling, target, proba)

        reg = ds_du.norm(dim=-1).mean() + ds_dv.norm(dim=-1).mean()

        return snake + self.reg_ratio*reg
