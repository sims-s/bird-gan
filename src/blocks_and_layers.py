import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


'''===================================General Layers==============================================='''



class EqualizedLRConv2d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride=1, padding=0):
        super(EqualizedLRConv2d, self).__init__()
        
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.weight = nn.Parameter(nn.init.normal_(torch.empty(channels_out, channels_in, kernel_size[0], kernel_size[1])))
        
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
        self.weight = nn.Parameter(nn.init.normal_(torch.empty(channels_in, channels_out, kernel_size[0], kernel_size[1])))
        
        self.stride = stride
        self.padding = padding
        
        self.bias = nn.Parameter(torch.FloatTensor(channels_out).fill_(0))
            
        self.scale_by = np.sqrt(2. / channels_in)
        
    def forward(self, x):
        return F.conv_transpose2d(input=x, weight=self.weight * self.scale_by, bias=self.bias,
                stride=self.stride, padding=self.padding)


class EqualizedLRLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(EqualizedLRLinear, self).__init__()
        self.weight = nn.Parameter(nn.init.normal_(nn.empty(dim_in, dim_out)))
        

        self.bias = nn.Parameter(torch.FloatTensor(dim_out).fill_(0))
            
        self.scale_by = sqrt(2./dim_in)
        
    def forward(self, x):
        return F.linear(x, self.weight * self.scale_by, self.bias)


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        
    def forward(self, x, eps=10**-8):
        norm = (x.pow(2).mean(dim=1,keepdim=True) + eps).sqrt()
        return x/norm

class AddStandardDeviation(nn.Module):
    def __init__(self):
        super(AddStandardDeviation, self).__init__()
        
    def forward(self, x):
        batch_size = x.size()[0]
        stds = x.std(dim=0).mean().view((1,1,1,1))
        stds = stds.repeat(batch_size, 1, x.size()[2], x.size()[3])
        return torch.cat([x, stds], 1)


'''===================================Progan Layers==============================================='''

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