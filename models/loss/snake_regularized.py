import torch.nn as nn
from math import log, exp

from .sliced_wasserstein_loss import SlicedWassersteinLoss

def sigmoid_alpha(T_max, reg_init, eps = 1e-4):
    gamma = (2/T_max)*log((1-eps)/eps)

    def f_alpha(x):
        a = exp(-gamma*(x-T_max/2))
        return 1 - reg_init*(a/(a+1))
    
    return f_alpha

def threshold_alpha(T_switch):
    def f_alpha(x):
        return 0 if x<T_switch else 1
    return f_alpha

def exponential_alpha(T_max, reg_init, eps = 1e-3):
    T = -T_max/log(eps)

    def f_alpha(x):
        return 1 - reg_init*exp(-x/T)
    
    return f_alpha


class ShapeReg(nn.Module):
    def __init__(self, shape, device, *args, **kwargs) -> None:
        super().__init__()
        self.shape = shape.unsqueeze(0).to(device)

    def forward(self,x):
        return (x-self.shape).norm(dim=-1).mean()

class RegularizedSnakeLoss(nn.Module):

    def __init__(self, reg_cps, device, f_alpha = "sigmoid", reg_part = 1.0,
                 epoch_reg_max = 200, eps = 1e-3, *args, **kwargs) -> None:
        super().__init__()

        self.shape_reg = ShapeReg(shape = reg_cps, device=device)

        self.device = device

        self.eps = eps

        self.snake_loss = SlicedWassersteinLoss(nb_angles=100, W_order=1, p_order=1, device=self.device)
        
        self.f_alpha = {"sigmoid" : sigmoid_alpha(epoch_reg_max, reg_part),\
                        "exponential" : exponential_alpha(epoch_reg_max, reg_part)}[f_alpha]
        
        self.f_epoch = 0
        self.f_epoch = self.update_reg_factor(0)
        
    def update_reg_factor(self, epoch):
        # stop computing the new value of the multiplication factor if it is close enough to 1
        if self.f_epoch == 1:
            return 1
        self.f_epoch = self.f_alpha(epoch)
        if self.f_epoch > 1 - self.eps:
            self.f_epoch = 1
        return self.f_epoch

    
    def forward(self, sampling, target, proba, cps):

        f = self.f_epoch
        snake = self.snake_loss(sampling, target, proba)
        if f > 1-self.eps:
            return snake
        reg = self.shape_reg(cps)
        return f*snake + (1-f)*reg


if __name__=="__main__":
    T_max = 50
    shift = 0.8
    f = sigmoid_alpha(T_max, shift)
    print(f"f(0) : {f(0)}, f(max) : {f(T_max)}")

    g = exponential_alpha(T_max, shift)
    print(f"g(0) : {g(0)}, g(max) : {g(T_max)}")