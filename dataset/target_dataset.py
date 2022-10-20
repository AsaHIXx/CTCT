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
import random
import os.path as osp
from utils.postprocessing import one_hot_encoder
# target_data = '/data/home/scv4524/run/xx/ANN/dataset/train_target_exp_2CT.txt'


class TargetClassAwareDataset(data.Dataset):
    def __init__(self, config, target_data, num_pclass, tgt_class_set, tgt_plabel_dict, num_steps=None, uk_list=None,
                 binary_label=None):
        self.num_pclass = num_pclass  # number of samples per class in each domain
        self.files = []
        labels = []
        self.num_steps = num_steps
        self.config = config
        self.ind2label = {}
        self.binary_label = binary_label
        self.tgt_files = {i: [] for i in tgt_class_set}
        self.label_set = tgt_class_set
        if uk_list is not None:
            self.uk_pool = list(uk_list.keys())

        for k, v in tgt_plabel_dict.items():
            if v in tgt_class_set:
                self.tgt_files[int(v)].append(k)
        self.target_data = torch.from_numpy(target_data)
    def __getitem__(self, index):
        if self.num_steps is not None:
            index = index % (len(self.tgt_files))

        label = self.label_set[index]
        tgt_pool = self.tgt_files[label]

        tgt_index = np.random.choice(len(tgt_pool), self.num_pclass)
        tgt_path = [tgt_pool[i] for i in tgt_index]

        tgt_labels = [label for i in tgt_index]
        tgt_labels = torch.Tensor(tgt_labels).long()

        tgt_imgs: list = []
        tgt_bi_labels = []
        for p in tgt_path:
            cur_img = self.target_data[p, :]
            tgt_imgs.append(cur_img)
            if self.binary_label is not None:
                bp_label = torch.Tensor([self.binary_label[p]])
            else:
                bp_label = torch.Tensor([0])
            tgt_bi_labels.append(bp_label)
        tgt_bi_labels = torch.stack(tgt_bi_labels).long()
        tgt_imgs = torch.stack(tgt_imgs, 0)
        if self.config.onehot:
            tgt_labels = one_hot_encoder(tgt_labels, self.config.num_classes)
        return tgt_imgs, tgt_labels, tgt_path, tgt_bi_labels

    def __len__(self):
        if self.num_steps is None:
            return len(self.tgt_files)
        else:
            return self.num_steps
