import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks_and_layers import *


'''===================================Progan Models==============================================='''

class ProGanGenerator(nn.Module):
    def __init__(self, depth, noise_size):
        super(ProGanGenerator, self).__init__()

        self.first_layer = FirstGenBlockProGan(noise_size)
        self.layers = nn.ModuleList()
        self.rgb_out_layers = nn.ModuleList([EqualizedLRConv2d(noise_size, 3, (1,1))])
        
        for i in range(depth - 1):
            if i <=2:
                self.layers.append(GenBlockProGan(noise_size, noise_size))
                self.rgb_out_layers.append(EqualizedLRConv2d(noise_size, 3, (1,1)))
            else:
                self.layers.append(GenBlockProGan(noise_size // 2**(i-3), noise_size // 2**(i-2)))
                self.rgb_out_layers.append(EqualizedLRConv2d(int(noise_size // 2**(i-2)), 3, (1,1)))
                
    def forward(self, x, depth, alpha):
        x = self.first_layer(x)
        # No fadein on the smallest size
        if depth==0:
            x = self.rgb_out_layers[0](x)
            return x
        else:
            # Run through all the layers except the one that we want to output
            for layer in self.layers[:depth-1]:
                x = layer(x)
            # upscale the learned output from the previous size and use that RGB output layer
            prev_layer = self.rgb_out_layers[depth-1](F.interpolate(x, scale_factor=2, mode='nearest'))
            # Then get the output for this layer
            this_layer = self.rgb_out_layers[depth](self.layers[depth-1](x))
            # return the convex combo of them
            return alpha * this_layer + (1-alpha)*prev_layer


class ProGanDiscriminator(nn.Module):
    def __init__(self, depth, noise_size):
        super(ProGanDiscriminator, self).__init__()
        
        self.last_block = DiscrimLastBlockProGan(noise_size)
        self.layers = nn.ModuleList([])
        self.from_rgb = nn.ModuleList([EqualizedLRConv2d(3, noise_size, (1,1))])
        
        for i in range(depth - 1):
            if i > 2:
                self.layers.append(DiscrimBlockProGan(noise_size // 2**(i-2), noise_size // 2**(i-3)))
                self.from_rgb.append(EqualizedLRConv2d(3, int(noise_size // 2**(i-2)), (1,1)))
            else:
                self.layers.append(DiscrimBlockProGan(noise_size, noise_size))
                self.from_rgb.append(EqualizedLRConv2d(3, noise_size, (1,1)))
                
    def forward(self, x, depth, alpha):
        if depth==0:
            x = self.from_rgb[0](x)
            x = self.last_block(x)
            return x
        else:
            smaller_features = self.from_rgb[depth-1](F.avg_pool2d(x, 2))
            features = self.layers[depth-1](self.from_rgb[depth](x))
            
            output = alpha * features + ((1-alpha) * smaller_features)
            for layer in reversed(self.layers[:depth-1]):
                output = layer(output)
            return self.last_block(output)
            