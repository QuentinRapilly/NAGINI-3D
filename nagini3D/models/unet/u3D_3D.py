from .unet_clean import UNetClean
import torch.nn as nn
from .internals.layers import cInstanceNorm2d, cBlockNorm3d, cBlockNorm3dSmooth, cBlockNorm3dSmoothV2

class DoubleConv(nn.Module):
    """
    [ Conv3d => BatchNorm (optional) => ReLU ] x 2
    """

    def __init__(self, in_ch: int, out_ch: int, train_bn: bool, inner_normalisation: str, padding_mode:str, **kwargs):
        super().__init__()
        inm = {'InstanceNorm':nn.InstanceNorm3d, 'BatchNorm':nn.BatchNorm3d,
                'None' : nn.Identity, 'cInstanceNorm2d' : cInstanceNorm2d,
                'cBlockNorm3d' :cBlockNorm3d, 'cBlockNorm3dSmooth' : cBlockNorm3dSmooth,
                'cBlockNorm3dSmoothV2': cBlockNorm3dSmoothV2}[inner_normalisation]
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode=padding_mode),
            inm(out_ch, track_running_stats=train_bn),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode=padding_mode),
            inm(out_ch, track_running_stats=train_bn),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)
    


class DoubleConvRes(nn.Module):
    """
    [ Conv3d => BatchNorm (optional) => ReLU ] x 2
    """

    def __init__(self, in_ch: int, out_ch: int, train_bn: bool, inner_normalisation: str, padding_mode:str, **kwargs):
        super().__init__()
        inm = {'InstanceNorm':nn.InstanceNorm3d, 'BatchNorm':nn.BatchNorm3d,
                'None' : nn.Identity, 'cInstanceNorm2d' : cInstanceNorm2d,
                'cBlockNorm3d' :cBlockNorm3d, 'cBlockNorm3dSmooth' : cBlockNorm3dSmooth,
                'cBlockNorm3dSmoothV2': cBlockNorm3dSmoothV2}[inner_normalisation]
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode=padding_mode),
            inm(out_ch, track_running_stats=train_bn),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode=padding_mode),
            inm(out_ch, track_running_stats=train_bn),
            nn.ReLU(inplace=True)
        )

        self.residual = nn.Conv3d(in_ch, out_ch, kernel_size=1, padding=0, padding_mode=padding_mode)

    def forward(self, x):
        return self.net(x) + self.residual(x)


class Down(nn.Module):
    """
    Downsampling (by Conv)
    """

    def __init__(self, in_ch: int, out_ch: int, scale_factor: int, padding_mode:str, **kwargs):
        super().__init__()
        self.in_ch = in_ch
        assert out_ch == in_ch, f'Expected out_ch = {in_ch} in Up got out_ch = {out_ch}'
        self.downsample = nn.Conv3d(in_ch, out_ch,
                                    kernel_size=scale_factor,
                                    stride=scale_factor,
                                    padding_mode=padding_mode)

    def forward(self, x):
        """Downsample input by scale factor

        Parameters
        ----------
        x : input tensor ( b, c, t, w, h)

        Returns
        -------
        x : output tensor (b, c, t, w // sf, h // sf)

        """
        assert x.shape[1] ==  self.in_ch, 'Incorrect number of channels'
        return self.downsample(x)

class Up(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    """

    def __init__(self, in_ch: int, out_ch: int, scale_factor: int, padding_mode:str,  bilinear : bool, **kwargs):
        super().__init__()
        assert out_ch == in_ch//scale_factor, f'Expected out_ch = {in_ch//scale_factor} in Up got out_ch = {out_ch}'
        self.upsample = None
        if padding_mode != 'zeros':
             print(f"Could not apply '{padding_mode}' in upsampling used padding_mode='zeros'")
        assert not bilinear, 'Bilinear upsampling not available for 3D Unet'
        # ValueError: Only "zeros" padding mode is supported for ConvTranspose3d
        self.upsample = nn.ConvTranspose3d(in_ch, out_ch,
                                           kernel_size=scale_factor,
                                           stride=scale_factor,
                                           padding_mode='zeros')

    def forward(self, x1):
        return self.upsample(x1)


class Unet3D_3D(UNetClean) :
    lalevels = { 'down' : Down,
                 'convl' : DoubleConv,
                 'transit' : nn.Identity,
                 'convr' : DoubleConv,
                 'up' : Up,
                 'final_layer' : nn.Conv3d}

    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
