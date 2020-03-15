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
            

'''===================================StyleGan Models==============================================='''
class StyleGanMapping(nn.Module):
    def __init__(self, latent_size_in, latent_size_out, num_layers, num_copies):
        super(StyleGanMapping, self).__init__()
        self.latent_size_in = latent_size_in
        self.latent_size_out = latent_size_out
        self.num_copies = num_copies

        layers = [PixelNorm()]
        for i in range(num_layers-1):
            layers.append(EqualizedLRLinear(latent_size_in, latent_size_in))
            layers.append(nn.LeakyReLU(.2))
        layers.append(EqualizedLRLinear(latent_size_in, latent_size_out))
        layers.append(nn.LeakyReLU(.2))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        if self.num_copies:
            x = x.unsqueeze(1).expand(-1, self.num_copies, -1).contiguous()
        return x

def num_feats_per_layer(layer_index, base=8192, decay=1, max_val=512):
    return min(int(base / (2 **((layer_index+1) * decay))), max_val)

class StyleGanGenSynthesis(nn.Module):
    def __init__(self, latent_size, max_depth):
        super(StyleGanGenSynthesis, self).__init__()

        self.max_depth = max_depth

        self.first_layer = FirstGenBlockStyleGan(num_feats_per_layer(0), latent_size)
        self.rgb_out_layers = nn.ModuleList([EqualizedLRConv2d(num_feats_per_layer(0), 3, (1,1))])
        self.layers = nn.ModuleList()

        for i in range(1, max_depth):
            prev_features = num_feats_per_layer(i-1)
            this_features = num_feats_per_layer(i)
            self.layers.append(GenBlockStyleGan(prev_features, this_features, latent_size))
            self.rgb_out_layers.append(EqualizedLRConv2d(this_features, 3, 1))

    def forward(self, latents, depth, alpha):
        x = self.first_layer(latents[:,:2])
        if depth > 0:
            for i, block in enumerate(self.layers[:depth-1]):
                x = block(x, latents[:, (i+1)*2:(i+1)*2+2])

            prev_depth_out = self.rgb_out_layers[depth-1](F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))
            this_depth_out = self.rgb_out_layers[depth](self.layers[depth-1](x, latents[:,2*depth:2*depth+2]))
            output = alpha * this_depth_out + (1-alpha) * prev_depth_out
            return output
        else:
            x =  self.rgb_out_layers[0](x)
            return x

class StyleGanGenerator(nn.Module):
    def __init__(self, max_depth, latent_size_in):
        super(StyleGanGenerator, self).__init__()
        self.max_depth = max_depth
        self.mapping_layers = StyleGanMapping(latent_size_in, latent_size_in, 8, max_depth*2)
        self.synthesis_layers = StyleGanGenSynthesis(latent_size_in, max_depth)

        self.truncation = Truncation(torch.zeros(latent_size_in))
        self.style_mix_prob = .9

    def forward(self, latents, depth, alpha):
        mapped_latents = self.mapping_layers(latents)
        if self.training:
            self.truncation.update(mapped_latents[0,0].detach())

            second_latents = torch.randn(latents.shape).to(latents.device)
            second_mapped_latents = self.mapping_layers(second_latents)
            # Have max_depth*2 style blocks
            layer_idx = torch.arange(self.max_depth * 2).view(1, -1, 1).to(latents.device)
            curr_layer = 2 * depth + 2
            cutoff = np.random.randint(1, curr_layer) if np.random.rand() < self.style_mix_prob else curr_layer

            mapped_latents = torch.where(layer_idx < cutoff, mapped_latents, second_mapped_latents)

            mapped_latents = self.truncation(mapped_latents)
        output = self.synthesis_layers(mapped_latents, depth, alpha).contiguous()
        return output


class StyleGanDiscriminator(nn.Module):
    def __init__(self, max_depth, latent_size):
        super(StyleGanDiscriminator, self).__init__()
        self.max_depth = max_depth

        self.layers = nn.ModuleList()
        self.from_rgb = nn.ModuleList()
        self.last_block = DiscrimLastBlockStyleGan(latent_size, latent_size)

        for i in range(self.max_depth-1, 0, -1):
            in_channels = num_feats_per_layer(i)
            out_channels = num_feats_per_layer(i-1)
            self.layers.append(DiscrimBlockStyleGan(in_channels, out_channels))
            self.from_rgb.append(EqualizedLRConv2d(3, in_channels, (1,1)))

        self.from_rgb.append(EqualizedLRConv2d(3, num_feats_per_layer(1), (1,1)))


    def forward(self, x, depth, alpha):
        if depth==0:
            x = self.from_rgb[-1](x)
            x = self.last_block(x)
            return x
        else:
            smaller_features = self.from_rgb[self.max_depth - depth](nn.AvgPool2d(2)(x))
            larger_features = self.layers[self.max_depth-depth-1](self.from_rgb[self.max_depth-depth-1](x))
            x = alpha * larger_features + (1-alpha)*(smaller_features)

            for layer in self.layers[(self.max_depth - depth):]:
                x = layer(x)

            x = self.last_block(x)
            return x




