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

def load_df_convert_to_dict(path, source):
    df = pd.read_csv(path + '%s/labels.csv'%source, usecols = ['id'])
    df['load_path'] = df['id'].apply(lambda x: path + '%s/images/%s.jpg'%(source, x))
    return df


def get_label_dict(data_path):
    sources = ['nabirds', 'cub', 'obd', 'flikr']
    sources = ['cub']
    dfs = [load_df_convert_to_dict(data_path, source) for source in sources]
    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    label_dict = df['load_path'].to_dict()
    return label_dict



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
            return np.moveaxis(batch, (1,2,3), (2,1,3))
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





def plot_gen_images(gen, depth, fade_in, noise_size, device):
    noise = generate_noise(16, noise_size, device=device)
    imgs = gen(noise, depth, fade_in).data.cpu().numpy()
    imgs = swap_channels_batch(imgs)
    
    fig, axs = plt.subplots(4,4)
    fig.set_size_inches(8,8)
    for i in range(4):
        for j in range(4):
            axs[i,j].imshow(imgs[4*i+j])
    plt.show()
    

def save_gen_fixed_noise(gen, depth, fade_in, counter, fixed_noise, save_path):
    this_save_path = save_path + 'imgs_fixed/'
    if not os.path.exists(this_save_path):
        os.mkdir(this_save_path)
    imgs = gen(fixed_noise, depth, fade_in).data.cpu().numpy()
        
    imgs = swap_channels_batch(imgs)
    fig, axs = plt.subplots(4,4)
    fig.set_size_inches(8,8)
    for i in range(4):
        for j in range(4):
            axs[i,j].imshow(imgs[4*i+j])
    plt.savefig(this_save_path + '%d_%d.png'%(depth, counter))
    plt.close(fig)


depth_to_img_size = lambda depth: 2**(depth+2)