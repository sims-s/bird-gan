import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



'''===================================General Layers==============================================='''
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
            # Other persons code doesn't scale by .25... seems like we should?
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
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3))
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
        return x

class NoiseLReluNormStyle(nn.Module):
    def __init__(self, nb_channels, latent_size):
        super(NoiseLReluNormStyle, self).__init__()
        self.layers = nn.Sequential(
            PerChannelNoise(nb_channels),
            nn.LeakyReLU(.2),
            nn.InstanceNorm2d(nb_channels)
        )

        self.style_mod = StyleMod(nb_channels, latent_size)

    def forward(self, x, other_style):
        x = self.layers(x)
        x = self.style_mod(x, other_style)
        return x

class Truncation(nn.Module):
    def __init__(self, avg_latent, max_layer=8, thresh=.7, ema_coeff=.995):
        super(Truncation, self).__init__()
        self.max_layer = max_layer
        self.thresh = thresh
        self.ema_coeff = ema_coeff
        self.register_buffer('avg_latent', avg_latent)
    
    def update(self, next_vec):
        self.avg_latent.copy_(self.ema_coeff * self.avg_latent + (1-self.ema_coeff) * next_vec)
    
    def forward(self, x):
        interp = torch.lerp(self.avg_latent, x, self.thresh)
        truncate_indexer = (torch.arange(x.size(1)) < self.max_layer).view(1, -1, 1).to(x.device)
        return torch.where(truncate_indexer, interp, x)


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

# what does gain mean and why is it here???
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





'''===================================Progan Blocks==============================================='''
class FirstGenBlockProGan(nn.Module):
    def __init__(self, noise_size):
        super(FirstGenBlockProGan, self).__init__()
        self.layers = nn.Sequential(
            EqualizedLRConv2dTranspose(noise_size, noise_size, (4,4)),
            nn.LeakyReLU(.2, inplace=True),
            EqualizedLRConv2d(noise_size, noise_size, (3,3), padding=1),
            nn.LeakyReLU(.2, inplace=True),
            PixelNorm(),
            
        )
    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        return self.layers(x)


class GenBlockProGan(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(GenBlockProGan, self).__init__()
        self.layers = nn.Sequential(
            EqualizedLRConv2d(channels_in, channels_out, (3,3), padding=1),
            nn.LeakyReLU(.2, inplace=True),
            PixelNorm(),
            EqualizedLRConv2d(channels_out, channels_out, (3,3), padding=1),
            nn.LeakyReLU(.2, inplace=True),
            PixelNorm(),
        )
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.layers(x)


class DiscrimLastBlockProGan(nn.Module):
    def __init__(self, channels_in):
        super(DiscrimLastBlockProGan, self).__init__()
        self.add_std = AddStandardDeviation()
        self.layers = nn.Sequential(
            EqualizedLRConv2d(channels_in + 1, channels_in, (3,3), padding=1),
            nn.LeakyReLU(.2, inplace=True),
            EqualizedLRConv2d(channels_in, channels_in, (4,4),),
            nn.LeakyReLU(.2, inplace=True),
            EqualizedLRConv2d(channels_in, 1, (1,1))
        )
    def forward(self, x):
        x = self.add_std(x)
        return self.layers(x).view(-1)


class DiscrimBlockProGan(nn.Module):
    def __init__(self, channels_in, chanels_out):
        super(DiscrimBlockProGan, self).__init__()
        self.layers = nn.Sequential(
            EqualizedLRConv2d(channels_in, channels_in, (3,3), padding=1),
            nn.LeakyReLU(.2, inplace=True),
            EqualizedLRConv2d(channels_in, chanels_out, (3,3), padding=1),
            nn.LeakyReLU(.2, inplace=True),
            nn.AvgPool2d(2)
        )
        
    def forward(self, x):
        return self.layers(x)


'''===================================Stylegan Blocks==============================================='''

class FirstGenBlockStyleGan(nn.Module):
    def __init__(self, nb_channels, latent_size):
        super(FirstGenBlockStyleGan, self).__init__()
        self.nb_channels = nb_channels
        self.const_input = nn.Parameter(torch.ones(1, nb_channels, 4, 4))
        self.bias = nn.Parameter(torch.ones(nb_channels))

        self.layers1 = NoiseLReluNormStyle(nb_channels, latent_size)
        self.conv = EqualizedLRConv2d(nb_channels, nb_channels, 3, padding=1)
        self.layers2 = NoiseLReluNormStyle(nb_channels, latent_size)

    def forward(self, latents):
        x = self.const_input.expand(latents.size(0), -1, -1, -1) + self.bias.view(1, -1, 1, 1)
        x = self.layers1(x, latents[:,0])
        x = self.conv(x)
        x = self.layers2(x, latents[:,1])
        return x

class GenBlockStyleGan(nn.Module):
    def __init__(self, channels_in, channels_out, latent_size):
        super(GenBlockStyleGan, self).__init__()

        self.conv1 = EqualizedConv2dUp(channels_in, channels_out, kernel_size=3)
        self.layers1 = NoiseLReluNormStyle(channels_out, latent_size)
        self.conv2 = EqualizedLRConv2d(channels_out, channels_out, kernel_size=3, padding=1)
        self.layers2 = NoiseLReluNormStyle(channels_out, latent_size)

    def forward(self, x, latents):
        print('Gen block input: ', x.size())
        x = self.conv1(x)
        x = self.layers1(x, latents[:,0])
        x = self.conv2(x)
        x = self.layers2(x, latents[:,1])
        print('Gen block output: ', x.size())
        return x
        
class DiscrimLastBlockStyleGan(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(DiscrimLastBlockStyleGan, self).__init__()
        self.first_layers = nn.Sequential(
            AddStandardDeviationGrouped(),
            EqualizedLRConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(.2)
        )
        self.second_layers = nn.Sequential(
            EqualizedLRLinear(in_channels * 4 * 4, mid_channels),
            nn.LeakyReLU(.2),
            EqualizedLRLinear(mid_channels, 1)
        )
    def forward(self, x):
        x = self.first_layers(x)
        x = x.view(x.size(0), -1)
        x = self.second_layers(x)
        return x


class DiscrimBlockStyleGan(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscrimBlockStyleGan, self).__init__()
        self.layers = nn.Sequential(
            EqualizedLRConv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(.2),
            BlurLayer(),
            EqualizedConv2dDown(in_channels, out_channels, kernel_size=3),
            nn.LeakyReLU(.2)
        )

    def forward(self, x):
        return self.layers(x)