import torch.nn as nn
from blocks_and_layers import *

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