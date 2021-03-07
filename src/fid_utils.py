import numpy as np
from scipy.linalg import sqrtm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader
import modeling_utils
from tqdm.auto import tqdm
from inception import InceptionV3
from shutil import rmtree

def get_inception(device):
    inception = InceptionV3(resize_input=False)
    inception.fc = nn.Identity()
    inception.to(device)
    return inception

# Calculate fid for images in a given path
def calculate_statistics_for_dataset(img_path, save_path, batch_size, device, max_batches=-1):
    inception = get_inception(device)
    transforms = T.Compose([
        T.Resize(299),
        T.ToTensor(),
    ])
    ds = modeling_utils.BirdDataset({i: img_path + f for i, f in 
                                enumerate(os.listdir(img_path))}, transform=transforms)
    all_features = []
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    if max_batches < 0:
        max_batches = len(loader)
    pbar = tqdm(total = max_batches, leave=False)
    for i, batch in enumerate(loader):
        if i > max_batches:
            break
        batch = batch.to(device)
        features = inception(batch)[0].data.cpu().numpy()
        all_features.append(features)
        pbar.update(1)
    pbar.close()
    
    all_features = np.vstack(all_features).squeeze()
    np.save(save_path, all_features)

# function doens't work sometimes? idk what's up with that
def calculate_fid(path1, path2):
    features1 = np.load(path1)
    features2 = np.load(path2)
    # print('loadded features')

    mean1 = np.mean(features1, axis=0)
    mean2 = np.mean(features2, axis=0)
    # print('computesd mead')
    cov1 = np.cov(features1, rowvar=False)
    cov2 = np.cov(features2, rowvar=False)
    # print('computed cov')


    diff = mean1 - mean2
    # print('computed diff')
    cov_mean, err = sqrtm(cov1.dot(cov2), disp=False)
    # print('computed cov')
    cov_mean = cov_mean.real
    # print('mad cov real')

    trace_cov_mean = np.trace(cov_mean) 
    # print('comptued trace')

    return diff.dot(diff) + np.trace(cov1) + np.trace(cov2) - 2*trace_cov_mean

def cleanup_fid(base_path):
    os.remove(base_path + 'fake.npy')
    for f in os.listdir(base_path + 'tmp_gen_images/'):
        os.remove(base_path + 'tmp_gen_images/' + f)



if __name__ == "__main__":
    # just for testing...
    calculate_statistics_for_dataset('../data/modeling_256/images/', '../fid_test/real_1.npy',
                8, torch.device('cuda'), 1024)
    calculate_statistics_for_dataset('../data/modeling_256/images/', '../fid_test/real_2.npy',
                8, torch.device('cuda'), 1024)

    print(calculate_fid('../fid_test/real_1.npy', '../fid_test/real_2.npy'))