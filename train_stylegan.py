import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

import os
import sys
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
import torchvision.transforms as transforms

from IPython.display import display

sys.path.append('./src/')
from modeling_utils import *
from blocks_and_layers import *
from models import *
from optimization_utils import *


def main():
    assert torch.cuda.is_available()

    NOISE_SIZE = 512
    GRAD_PEN_WEIGHT = 10
    MAX_DEPTH = 8
    FADE_IN_PCT = .5
    BATCH_SIZE = 128
    NB_EPOCHS = 5
    SAMPLE_INTERVAL = 3
    IMG_SIZE = 256
    BATCH_SIZES = [1, 1, 1, 1, 1, 1, 1]

    load_path = None

    data_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])

    sizes = [2**i for i in range(2, int(1+np.log(IMG_SIZE) / np.log(2)))]
    BATCH_SIZES = {size : bs for size, bs in zip(sizes, BATCH_SIZES)}
    label_dicts = {size : get_label_dict('data/modeling_%d/'%size) for size in sizes}
    datasets = {}
    for k, v in label_dicts.items():
        datasets[k] = BirdDataset(v, transform=data_transform)
    data_loaders = {}
    for k, v in label_dicts.items():
        data_loaders[k] = torch.utils.data.DataLoader(datasets[k], batch_size=BATCH_SIZES[k], num_workers=0, shuffle=True)
    
    device = torch.device('cuda')

    if not load_path:
        FIXED_NOISE = generate_noise(16, NOISE_SIZE, device)

        gen = StyleGanGenerator(MAX_DEPTH, NOISE_SIZE).to(device)
        gen_opt = optim.Adam([{'params': gen.mapping_layers.parameters(), 'lr':1e-5}, 
                            {'params':gen.synthesis_layers.parameters()},
                            {'params':gen.truncation.parameters()}], betas=(0, .99))
        gen_ema = copy.deepcopy(gen)
        update_average(gen_ema, gen, 0)

        discrim = StyleGanDiscriminator(MAX_DEPTH, NOISE_SIZE).to(device)
        discrim_opt = optim.Adam(discrim.parameters(), betas=(0, .99))

        save_path = './' + datetime.datetime.now().strftime('%m-%d_%H-%M') + '/'

        # train_on_depth_wasserstein_gp(gen, gen_opt, gen_ema, discrim, discrim_opt, 0, IMG_SIZE, NB_EPOCHS, FADE_IN_PCT, NOISE_SIZE, 
        #               GRAD_PEN_WEIGHT, data_loaders[4], device, FIXED_NOISE, save_path, sample_interval=8, save_samples=True)

        # NB_EPOCHS = 10

        # train_on_depth_wasserstein_gp(gen, gen_opt, gen_ema, discrim, discrim_opt, 1, IMG_SIZE, NB_EPOCHS, FADE_IN_PCT, NOISE_SIZE, 
        #               GRAD_PEN_WEIGHT, data_loaders[8], device, FIXED_NOISE, save_path, sample_interval=8, save_samples=True)
        

        # train_on_depth_wasserstein_gp(gen, gen_opt, gen_ema, discrim, discrim_opt, 2, IMG_SIZE, NB_EPOCHS, FADE_IN_PCT, NOISE_SIZE, 
        #               GRAD_PEN_WEIGHT, data_loaders[16], device, FIXED_NOISE, save_path, sample_interval=8, save_samples=True)

        # train_on_depth_wasserstein_gp(gen, gen_opt, gen_ema, discrim, discrim_opt, 3, IMG_SIZE, NB_EPOCHS, FADE_IN_PCT, NOISE_SIZE, 
        #               GRAD_PEN_WEIGHT, data_loaders[32], device, FIXED_NOISE, save_path, sample_interval=8, save_samples=True)

        # train_on_depth_wasserstein_gp(gen, gen_opt, gen_ema, discrim, discrim_opt, 4, IMG_SIZE, NB_EPOCHS, FADE_IN_PCT, NOISE_SIZE, 
        #               GRAD_PEN_WEIGHT, data_loaders[64], device, FIXED_NOISE, save_path, sample_interval=8, save_samples=True)

        train_on_depth_wasserstein_gp(gen, gen_opt, gen_ema, discrim, discrim_opt, 5, IMG_SIZE, NB_EPOCHS, FADE_IN_PCT, NOISE_SIZE, 
                      GRAD_PEN_WEIGHT, data_loaders[128], device, FIXED_NOISE, save_path, sample_interval=8, save_samples=True)

        train_on_depth_wasserstein_gp(gen, gen_opt, gen_ema, discrim, discrim_opt, 6, IMG_SIZE, NB_EPOCHS, FADE_IN_PCT, NOISE_SIZE, 
                      GRAD_PEN_WEIGHT, data_loaders[256], device, FIXED_NOISE, save_path, sample_interval=8, save_samples=True)




if __name__=="__main__":
    main()