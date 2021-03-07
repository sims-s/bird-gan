import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

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
            return batch.permute(0,2,3,1).contiguous()

def generate_noise(size, noise_size, device, seed=None):
    if seed:
        torch.manual_seed(seed)
    noise = torch.randn(size, noise_size, device=device)
    return noise

def sample_gen_images(gen, noise_size, device, noise=None, batch_size=16, **kwargs):
    if noise is None:
        noise = generate_noise(batch_size, noise_size, device=device)
    imgs = gen(noise, **kwargs).data.cpu().numpy()
    imgs = swap_channels_batch(imgs)
    imgs = post_model_process(imgs)
    return imgs

def post_model_process(imgs):
    if torch.is_tensor(imgs):
        imgs = imgs.data.cpu().numpy()
    min_val = np.min(imgs, axis=(1,2,3)).reshape((-1,1,1,1))
    max_val = np.max(imgs, axis=(1,2,3)).reshape((-1,1,1,1))
    imgs = (imgs - min_val) / (max_val - min_val)
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
    # if not os.path.exists(save_path + 'fixed_noise.npy'):
    #     np.save(save_path + 'fixed_noise.npy', fixed_noise.data.cpu().numpy())

    imgs = gen(fixed_noise, **kwargs).data.cpu().numpy()
    imgs= np.clip(imgs, 0, 1)
        
    imgs = swap_channels_batch(imgs)
    fig, axs = plt.subplots(4,4)
    fig.set_size_inches(8,8)
    for i in range(4):
        for j in range(4):
            axs[i,j].imshow(imgs[4*i+j])
    # plt.savefig(save_path + 'step_%d.png'%(save_idx))
    plt.savefig(save_path + '%d.png'%(save_idx))
    plt.close(fig)


def save_gen_fid_images(gen, noise_size, fid_dir, n_fid_samples, device, **kwargs):
    batch_size = 16
    nb_batches = n_fid_samples//batch_size + int((n_fid_samples % batch_size)==0)
    pbar = tqdm(total = nb_batches, leave=False)
    counter = 0
    for i in range(nb_batches):
        imgs = sample_gen_images(gen, noise_size, device, batch_size=batch_size, **kwargs)
        for img in imgs:
            pil_img = Image.fromarray((img*255).astype(np.uint8))
            pil_img.save(fid_dir + "tmp_gen_images/" + '%d.jpg'%counter, quality=95)
            counter += 1
        pbar.update(1)
    pbar.close()

depth_to_img_size = lambda depth: 2**(depth+2)