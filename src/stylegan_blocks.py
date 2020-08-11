import torch
import torch.nn as nn
from blocks_and_layers import *

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
        x = self.conv1(x)
        x = self.layers1(x, latents[:,0])
        x = self.conv2(x)
        x = self.layers2(x, latents[:,1])
        return x
        
class DiscrimLastBlockStyleGan(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(DiscrimLastBlockStyleGan, self).__init__()
        self.first_layers = nn.Sequential(
            AddStandardDeviationGrouped(4),
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