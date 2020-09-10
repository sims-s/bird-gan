import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EqualizedConv2dUp(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride=1, intermediate=None):
        super(EqualizedConv2dUp, self).__init__()
        self.upscale = Upscale2d()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.scale_by = np.sqrt(2 / (self.kernel_size[0]*self.kernel_size[1]*channels_in))

        self.weight = nn.Parameter(torch.randn(channels_out, channels_in, self.kernel_size[0], self.kernel_size[1]))
        self.bias = nn.Parameter(torch.zeros(channels_out))
        self.intermediate = intermediate

    def forward(self, x):
        if min(x.shape[2:]) >= 64:
            scaled_weights = self.weight * self.scale_by
            scaled_weights = scaled_weights.permute(1,0,2,3)
            scaled_weights = F.pad(scaled_weights, [1,1,1,1])
            # Other code doesn't multiply by .25, seems like we should
            scaled_weights = (scaled_weights[:, :, 1:, 1:] + scaled_weights[:, :, :-1, 1:] + \
                            scaled_weights[:, :, 1:, :-1] + scaled_weights[:, :, :-1, :-1]) * .25
            x = F.conv_transpose2d(x, scaled_weights, stride=2, padding=(self.weight.size(-1)-1)//2)

        else:
            x = self.upscale(x)
            x = F.conv2d(x, self.weight * self.scale_by, None, padding = self.kernel_size[0]//2)
        
        if self.intermediate is not None:
            x = self.intermediate(x)
        x = x + self.bias.view(1, -1, 1, 1)
        return x

class EqualizedConv2dDown(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride=1):
        super(EqualizedConv2dDown, self).__init__()
        self.downscale = Downscale2d()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.scale_by = np.sqrt(2 / (self.kernel_size[0]*self.kernel_size[1]*channels_in))
        
        self.weight = nn.Parameter(torch.randn(channels_out, channels_in, self.kernel_size[0], self.kernel_size[1]))
        self.bias = nn.Parameter(torch.zeros(channels_out))
        

    def forward(self, x):
        # why are these different size conditions?
        if min(x.shape[2:]) >= 64:
            scaled_weights = self.weight * self.scale_by
            scaled_weights = F.pad(scaled_weights, [1,1,1,1])
            scaled_weights = (scaled_weights[:, :, 1:, 1:] + scaled_weights[:, :, :-1, 1:] + \
                              scaled_weights[:, :, 1:, :-1] + scaled_weights[:, :, :-1, :-1]) * .25
            x = F.conv2d(x, scaled_weights, stride=2, padding=(scaled_weights.size(-1)-1)//2)
        else:
            x = F.conv2d(x, self.weight*self.scale_by, None, padding=self.kernel_size[0]//2)
            x = self.downscale(x)
        x = x + self.bias.view(1, -1, 1, 1)
        return x


class PerChannelNoise(nn.Module):
    def __init__(self, channels):
        super(PerChannelNoise, self).__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3)).to(x.device)
        else:
            if not hasattr(self, 'noise'):
                raise ValueError('Need to either pass or set noise!')
            nosie = self.noise

        return x + self.weight.view(1, -1, 1, 1) * noise



class EqualizedLRConv2d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride=1, padding=0):
        super(EqualizedLRConv2d, self).__init__()
        
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(channels_out, channels_in, kernel_size[0], kernel_size[1]))
        
        self.stride = stride
        self.padding = padding
        
        self.bias = nn.Parameter(torch.FloatTensor(channels_out).fill_(0))
        
        self.scale_by = np.sqrt(2. / (kernel_size[0]*kernel_size[1]*channels_in))
        
    
    def forward(self, x):
        return F.conv2d(input=x, weight=self.weight * self.scale_by, bias=self.bias,
                        stride=self.stride, padding=self.padding)
        
        
class EqualizedLRConv2dTranspose(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride=1, padding=0):
        super(EqualizedLRConv2dTranspose, self).__init__()
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(channels_in, channels_out, kernel_size[0], kernel_size[1]))
        
        self.stride = stride
        self.padding = padding
        
        self.bias = nn.Parameter(torch.zeros(channels_out).fill_(0))
            
        self.scale_by = np.sqrt(2. / channels_in)
        
    def forward(self, x):
        return F.conv_transpose2d(input=x, weight=self.weight * self.scale_by, bias=self.bias,
                stride=self.stride, padding=self.padding)


class EqualizedLRLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(EqualizedLRLinear, self).__init__()
        self.scale_by = np.sqrt(2./dim_in)
        self.weight = nn.Parameter(torch.randn(dim_out, dim_in))
        self.bias = nn.Parameter(torch.zeros(dim_out))   
        
        
    def forward(self, x):
        return F.linear(x, self.weight * self.scale_by, self.bias)


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        
    def forward(self, x, eps=10**-8):
        norm = x.pow(2).mean(dim=1,keepdim=True) + eps
        return x * torch.rsqrt(norm)

class AddStandardDeviation(nn.Module):
    def __init__(self):
        super(AddStandardDeviation, self).__init__()
        
    def forward(self, x):
        batch_size = x.size()[0]
        stds = x.std(dim=0).mean().view((1,1,1,1))
        stds = stds.repeat(batch_size, 1, x.size()[2], x.size()[3])
        return torch.cat([x, stds], 1)

class AddStandardDeviationGrouped(nn.Module):
    def __init__(self, group_size=4):
        super(AddStandardDeviationGrouped, self).__init__()
        self.group_size = group_size

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        group_size = min(self.group_size, batch_size)
        if batch_size % group_size:
            group_size = batch_size
        y = x.reshape([group_size, -1, 1, channels, height, width])
        y = y - y.mean(0, keepdim=True)
        y = ((y**2).mean(0, keepdim=True) + 10**-8).sqrt()
        y = y.mean([3,4,5], keepdim=True).squeeze(3)
        y = y.expand(group_size, -1, -1, height, width).clone().reshape(batch_size, 1, height, width)
        return torch.cat([x, y], dim=1)


class StyleMod(nn.Module):
    def __init__(self, nb_channels, latent_size):
        super(StyleMod, self).__init__()
        self.layer = EqualizedLRLinear(latent_size, nb_channels*2)

    def forward(self, x, latent):
        style = self.layer(latent)
        style = style.view(-1, 2, x.size(1), 1, 1)
        x = x + x * style[:,0] + style[:,1]
        return x.contiguous()

# Upscale/downscale come directly from here: https://github.com/huangzh13/StyleGAN.pytorch/
class Upscale2d(nn.Module):
    @staticmethod
    def upscale2d(x, factor=2, gain=1):
        assert x.dim() == 4
        if gain != 1:
            x = x * gain
        if factor != 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, factor, -1, factor)
            x = x.contiguous().view(shape[0], shape[1], factor * shape[2], factor * shape[3])
        return x

    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        return self.upscale2d(x, factor=self.factor, gain=self.gain)

class Downscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor
        self.gain = gain
        if factor == 2:
            f = [np.sqrt(gain) / factor] * factor
            self.blur = BlurLayer(kernel=f, normalize=False, stride=factor)
        else:
            self.blur = None

    def forward(self, x):
        assert x.dim() == 4
        # 2x2, float32 => downscale using _blur2d().
        if self.blur is not None and x.dtype == torch.float32:
            return self.blur(x)

        # Apply gain.
        if self.gain != 1:
            x = x * self.gain

        # No-op => early exit.
        if self.factor == 1:
            return x

        return F.avg_pool2d(x, self.factor)


class BlurLayer(nn.Module):
    def __init__(self, kernel=None, normalize=True, flip=False, stride=1):
        super(BlurLayer, self).__init__()
        if kernel is None:
            kernel = [1, 2, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer('kernel', kernel)
        self.stride = stride

    def forward(self, x):
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=int((self.kernel.size(2) - 1) / 2),
            groups=x.size(1)
        )
        return x