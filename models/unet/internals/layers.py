import torch.nn as nn
import torch
import einops

class cInstanceNorm2d(nn.Module):

    def __init__(self, num_features, affine=False, track_running_stats=False, epsilon=1e-05) :
        """
        Applies Instance Normalisation on the 2 last dimensions of a N-d input.
        .. math::
            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
        """
        super().__init__()
        assert track_running_stats == False, f'track_running_stats = {track_running_stats} not implemented'
        assert affine == False, f'track_running_stats = {affine} not implemented'

        self.eps = epsilon
        self.num_features = num_features

    def forward(self, x) :
        """
        Applies layer normalisation.

        Args :
            x (N, C, *, H, W) : applies layer normalisation over (H, W).
                                Each channels (N, C, *) is independent
        Returns :
            normalised x (N, C, *, H, W)
        """
        assert x.shape[1] == self.num_features,\
         f'Expected (N, {self.num_features}, ...), got {x.shape}'

        r = (x - x.mean(axis=(-2, -1), keepdim=True))
        return r / torch.sqrt(x.var(axis=(-2, -1), keepdim=True, unbiased=False) + self.eps)

class cBlockNorm3d(nn.Module):

    def __init__(self, num_features, block_size=9, affine=False, track_running_stats=False, epsilon=1e-05) :
        """
        Normalise spatio temporal block with blocks of block_size dimension
        .. math::
            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
        """
        super().__init__()
        assert track_running_stats == False, f'track_running_stats = {track_running_stats} not implemented'
        assert affine == False, f'track_running_stats = {affine} not implemented'

        self.eps = epsilon
        self.num_features = num_features
        self.block_size = block_size

    def forward(self, x) :
        """
        Applies layer normalisation.

        Args :
            x (N, C, T, H, W) : applies layer normalisation over (block_size, H, W).
                                Each channel (N, C, *) is and each block is
        Returns :
            normalised x (N, C, T, H, W)
        """
        assert x.shape[1] == self.num_features,\
         f'Expected (N, {self.num_features}, ...), got {x.shape}'
        t = x.shape[2]
        if t % self.block_size != 0 :
            p = (self.block_size - t % self.block_size)
            x =nn.functional.pad(x, (0,0,0,0,p // 2 , p// 2+ p%2), 'reflect')
        num_groups = (x.shape[2] // self.block_size)
        x = einops.rearrange(x, 'b c t i j -> (b c) t i j', c=self.num_features)
        x = nn.functional.group_norm(x, num_groups)
        x = einops.rearrange(x, '(b c) t i j -> b c t i j', c=self.num_features)
        if t % self.block_size != 0 :
            x = x[:,:,p//2:-(p// 2+ p%2)]
        return x


class cBlockNorm3dSmooth(nn.Module):

    def __init__(self, num_features, block_size=9, affine=False, track_running_stats=False, epsilon=1e-05) :
        """
        Smooth normalisation on temporal dimension
        .. math::
            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
        """
        super().__init__()
        assert track_running_stats == False, f'track_running_stats = {track_running_stats} not implemented'

        self.affine = affine
        self.eps = epsilon
        self.num_features = num_features
        self.block_size = block_size
        self.conv_gaussian = nn.Conv1d(num_features,num_features, block_size,
                                       stride=1, padding='same',
                                       padding_mode='reflect',
                                       bias=False, groups=num_features)
        kernel = self.get_gaussian_kernel(block_size)
        self.conv_gaussian.weight.data = einops.repeat(kernel, 'ks -> c 1 ks', c=num_features).clone()
        self.conv_gaussian.weight.requires_grad_(False)
        if self.affine :
            self.weight =  torch.nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
            self.bias =  torch.nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))

    @staticmethod
    def get_gaussian_kernel(block_size, std=20) :
        #kernel = (-torch.arange(-(block_size+1)//2+1, (block_size+1)//2)**2 / (2*std)).exp()
        kernel = torch.ones(block_size)
        kernel /= kernel.sum()
        return kernel

    def forward(self, x) :
        """
        Applies layer normalisation.

        Args :
            x (N, C, T, H, W) : applies layer normalisation over (block_size, H, W).
                                Each channel (N, C, *) is and each block is
        Returns :
            normalised x (N, C, T, H, W)
        """
        assert x.shape[1] == self.num_features,\
         f'Expected (N, {self.num_features}, ...), got {x.shape}'
        # Apparently we don't need to detach the gradients
        # https://discuss.pytorch.org/t/batchnorm-and-back-propagation/65765/4
        # Step 1 : Remove running mean
        xmean = x.mean(axis=(-2,-1))
        x = (x - self.conv_gaussian(xmean)[...,None, None])

        # Step 2 : Remove running variance
        xvar = x.var(axis=(-2,-1))
        x_n = (x / torch.sqrt(self.conv_gaussian(xvar)[...,None, None]+self.eps))

        if self.affine :
            x = x * self.weight + self.bias
        return x


class cBlockNorm3dSmoothV2(nn.Module):

    def __init__(self, num_features, block_size=9, affine=False, track_running_stats=False, epsilon=1e-05) :
        """
        Smooth normalisation on temporal dimension
        .. math::
            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
        """
        super().__init__()
        assert track_running_stats == False, f'track_running_stats = {track_running_stats} not implemented'
        assert affine == False, f'affine = {affine} not implemented'


        self.affine = affine
        self.eps = epsilon
        self.num_features = num_features
        self.block_size = block_size
        if self.affine :
            self.weight =  torch.nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
            self.bias =  torch.nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))

    def forward(self, x) :
        """
        Applies layer normalisation.

        Args :
            x (N, C, T, H, W) : applies layer normalisation over (block_size, H, W).
                                Each channel (N, C, *) is and each block is
        Returns :
            normalised x (N, C, T, H, W)
        """
        assert x.shape[1] == self.num_features,\
         f'Expected (N, {self.num_features}, ...), got {x.shape}'
        # Apparently we don't need to detach the gradients
        # https://discuss.pytorch.org/t/batchnorm-and-back-propagation/65765/4
        # Step 1 : Remove running mean
        x_f = nn.functional.pad(x, (0,0,0,0, self.block_size  // 2 , self.block_size  // 2), 'reflect')
        x_f = x_f.unfold(2, self.block_size, 1)
        mu = einops.reduce(x_f, 'b c t i j k -> b c t 1 1', 'mean')
        std = (einops.reduce(x_f**2, 'b c t i j k -> b c t 1 1', 'mean') - mu**2 + 1e-8).sqrt()
        return (x - mu)/std
