import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

import os
import sys

import imageio as io
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BirdDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, transform=None):
        self.data_dict = data_dict
        self.transform = transform
        
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        img = Image.open(self.data_dict[idx])
        if self.transform:
            img = self.transform(img)
        return img
    

def swap_channels_batch(batch):
    if isinstance(batch, np.ndarray):
        if batch.shape[3]==3:
            # batch size, width, height, channels
            return np.moveaxis(batch, (1,2,3), (2,3,1))
        else:
            # batch size, channels, widht height
            return np.moveaxis(batch, (1,2,3), (3,1,2))
    elif isinstance(batch, torch.Tensor):
        if batch.size()[3]==3:
            return batch.permute(0,3,1,2).contiguous()
        else:
            return batch.permute(0,2,1,3).contiguous()

def generate_noise(size, noise_size, device):
    noise = torch.randn(size, noise_size, device=device)
    return noise

def sample_gen_images(gen, noise_size, device, **kwargs):
    noise = generate_noise(16, noise_size, device=device)
    imgs = gen(noise, **kwargs).data.cpu().numpy()
    imgs = swap_channels_batch(imgs)
    imgs = np.clip(imgs, 0, 1)
    return imgs

def plot_imgs(imgs):
    fig, axs = plt.subplots(4, 4)
    fig.set_size_inches(8, 8)
    for i in range(4):
        for j in range(4):
            axs[i,j].imshow(imgs[4*i+j])
    plt.show()

def save_imgs(imgs, save_path):
    fig, axs = plt.subplots(4,4)
    fig.set_size_inches(8,8)
    for i in range(4):
        for j in range(4):
            axs[i,j].imshow(imgs[4*i+j])
    plt.savefig(save_path)
    plt.close(fig)


def save_gen_fixed_noise(gen, fixed_noise, save_path, save_idx, **kwargs):
    gen.eval()
    if not os.path.exists(save_path + 'fixed_noise.npy'):
        np.save(save_path + 'fixed_noise.npy', fixed_noise.data.cpu().numpy())

    imgs = gen(fixed_noise, **kwargs).data.cpu().numpy()
    imgs= np.clip(imgs, 0, 1)
        
    imgs = swap_channels_batch(imgs)
    fig, axs = plt.subplots(4,4)
    fig.set_size_inches(8,8)
    for i in range(4):
        for j in range(4):
            axs[i,j].imshow(imgs[4*i+j])
    plt.savefig(save_path + 'step_%d.png'%(save_idx))
    plt.close(fig)

depth_to_img_size = lambda depth: 2**(depth+2)