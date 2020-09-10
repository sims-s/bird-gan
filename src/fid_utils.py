import numpy as np
import torch.nn.functional as F
from torchvision import models, datasets, transforms as T


def calculate_fid_for_dataset(path):
    models.inception_v3(pretrained=True)
    