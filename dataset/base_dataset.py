import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import imageio
from utils.postprocessing import one_hot_encoder


class BaseDataset(data.Dataset):
    def __init__(self, exp, label, config):
        self.exp = exp
        self.label = label
        self.config = config

    def __getitem__(self, index):
        name = index
        img = self.exp[index, :]
        label = self.label[index]
        if self.config.onehot:
            label = one_hot_encoder(label, self.config.num_classes)
        return img, label, name

    def __len__(self):
        return len(self.label)
