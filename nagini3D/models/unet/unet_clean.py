import torch
from torch import nn as nn
from torch.nn import functional as F
from functools import partial


class Level(nn.Module) :
    def __init__(self, ft_in, *, fst_c=None, down, scale_factor=2, convl, transit, convr, up, **kwargs) :
        super().__init__()
        self.down = down(in_ch = ft_in, out_ch = ft_in, scale_factor = scale_factor, **kwargs)
        self.convl = convl(ft_in if fst_c is None else fst_c, ft_in*2, **kwargs)
        self.transit = transit(ft_in*2, ft_in*2)
        self.convr = convr(ft_in*4, ft_in*2, **kwargs)
        self.up = up(ft_in*2, ft_in, scale_factor=scale_factor, **kwargs)

    def forward_down(self, x) :
        """Run the downard phase of the given level
        Parameters
        ----------
            x : input from previous level ( b, c, w, h)

        Returns
        -------
            x : output for next level ( b, c*2, w // sf, h // sf)
            skip : skip connection for same level ( b, c*2, w // sf, h /sf)
        """
        x = self.down(x)
        x = self.convl(x)
        skip = self.transit(x)
        return x, skip

    def forward_up(self, x, skip) :
        """Run the downard phase of the given level
        Parameters
        ----------
            x : input from next level ( b, c*2, w // sf, h // sf)
            skip : skip connection for same level ( b, c*2, w // sf, h // sf)

        Returns
        -------
            x : output for previous level ( b, c, w, h)
        """
        if skip is None :
            o = x
        else :
            x = self.shape_adapt(x, skip.shape)
            o =  torch.cat([skip, x], axis=1)
        o = self.convr(o)
        return self.up(o)


    @staticmethod
    def shape_adapt(x, shape_skip):
        """
        Adaptation of the shape of x if necessary ( due to non even feature map shape)
        """
        pads = []
        for i in range(2, len(x.shape))[::-1] : # Pad from right to left
            diff = shape_skip[i] - x.shape[i]
            pads.extend([diff//2, diff - diff//2])
        return F.pad(x, pads)

class UNetClean(nn.Module):
    """
    Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
    <https://arxiv.org/abs/1505.04597>`_

    Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox

    Args:
        num_classes: Number of output classes required
        input_channels: Number of channels in input images (default 3)
        num_layers: Number of layers in each side of U-net (default 5)
        features_start: Number of features in first layer (default 64)
        kwargs :
            bilinear: Whether to use bilinear interpolation or transposed convolutions (default) for upsampling.
            train_bn : Whether to use accumulated batch parameters ( "trained" ) or per batch values
            inner_normalisation : Type of normalisation to use ['InstanceNorm', 'BatchNorm', 'None']
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        num_layers: int = 5,
        features_start: int = 64,
        **kwargs
    ):


        super().__init__()
        mLevel = partial(Level, **self.lalevels)
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.features_start = features_start
        print(f'Num layers : {self.num_layers} Features Start : {features_start} Padding Mode : {kwargs["padding_mode"]}')
        assert features_start % 2 == 0, 'Feature start need to be even'
        l = {}
        l['level_0'] = mLevel(features_start // 2, fst_c=input_channels, down=nn.Identity, up=nn.Identity, **kwargs) # first level
        for k in range(1, num_layers-1):
            l[f'level_{k}'] = mLevel(features_start * 2**(k-1), **self.lalevels, **kwargs)
        l[f'level_{num_layers-1}'] = mLevel(features_start * 2**(num_layers-2), convr=nn.Identity,
                                      transit=self.lalevels.get('transit_bottleneck',self.lalevels['transit']), **kwargs) # last level

        l['final_layer'] = self.lalevels['final_layer'](features_start, num_classes, kernel_size=1) # Final Output

        self.layers = nn.ModuleDict(l)

    def forward(self, x):
        """
        Params :
            x : model input ( b, c, I, J)
        Returns:
            Segmentation : model segmentation ( b, L, I , J)
            Hidden Features : list with skip hidden representation ( num_layers, b, Ls, Is, Js)
        """
        bottleneck, skips = self.encode(x)
        x = self.decode(bottleneck, skips)
        out = self.layers['final_layer'](x)
        hidden_feats = skips + [bottleneck]

        return out, {'HiddenV' : hidden_feats}

    def encode(self, x) :
        """"
        Encode the input x and returns the bottleneck and skips.
        Each skip is the result of a "downsampling" and "transit" operation
        Params :
            x : model input ( b, c, I, J)
        Returns :
            bottleneck (b, Ls, Is, Js) : Bottom feature map
            skips : [b, Lk, Ik, Jk]*(num_layers-1) : skip connections
        """
        # Top to Bottom
        skips = []
        for k in range(self.num_layers-1) :
            x , skip = self.layers[f'level_{k}'].forward_down(x)
            skips.append(skip)
        _, bottleneck = self.layers[f'level_{self.num_layers-1}'].forward_down(x)
        return bottleneck, skips

    def decode(self, bottleneck, skips) :
        """
        Decode starting from hidden feats. Use the given skip connections
        Params :
            bottleneck (b, Ls, Is, Js) : Bottom feature map
            skips : [b, Lk, Ik, Jk]*(num_layers-1) : skip connections
        Returns :
            x (b, L_ft, I , J) : final feature map
        """
        x = self.layers[f'level_{self.num_layers-1}'].forward_up(bottleneck,None)
        for k in range(self.num_layers-1)[::-1] :
            x = self.layers[f'level_{k}'].forward_up(x, skips[k])
        return x

    def get_output_shape(self, in_dim) :
        """
        Return output shapes for this unet structure
        params : in_dim : input dimension ( c, w, h)
        """
        fk_in = torch.zeros((1, in_dim[0], in_dim[1], in_dim[2]))
        output, hidden_feats = self.forward(fk_in)
        return output.shape, hidden_feats.shape

    @staticmethod
    def add_specific_args(parser, prefix=''):
        #parser = ArgumentParser(parents=[parent_parser], add_help=False)
        prefix += 'unet.'
        parser.add_argument(f'--{prefix}train_bn', action='store_true')
        parser.add_argument(f'--{prefix}num_layers', type=int, default=3)
        parser.add_argument(f'--{prefix}features_start', type=int, default=32)
        parser.add_argument(f'--{prefix}padding_mode', type=str, choices=['zeros', 'reflect'], default='reflect')
        parser.add_argument(f'--{prefix}bilinear', action='store_true', help='bilinear upsampling in unet instead of convtranspose')
        parser.add_argument(f'--{prefix}inner_normalisation', '-inm', type=str, choices=['InstanceNorm', 'BatchNorm', 'None',
                                                                                     'cInstanceNorm2d', 'cBlockNorm3d',
                                                                                     'cBlockNorm3dSmooth',
                                                                                     'cBlockNorm3dSmoothV2'], default='InstanceNorm')

        return parser
