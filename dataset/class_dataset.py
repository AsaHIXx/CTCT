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


class ClassAwareDataset(data.Dataset):
    def __init__(self, config, source_data, target_data, source_label, num_pclass, src_class_set, tgt_class_set, tgt_plabel_dict,
                 num_steps=None, uk_list=None):
        self.num_pclass = num_pclass  # number of samples per class in each domain
        self.files = []
        # self.transform = transform
        labels = []
        self.num_steps = num_steps
        assert src_class_set == tgt_class_set
        # if len(src_class_set) == 0:
        self.config = config
        self.ind2label = {}
        for i in range(len(src_class_set)):
            self.ind2label[i] = src_class_set[i]

        self.src_files = {i: [] for i in src_class_set}
        self.tgt_files = {i: [] for i in tgt_class_set}
        self.label = torch.from_numpy(source_label)
        self.src_data = torch.from_numpy(source_data)
        self.target_data = torch.from_numpy(target_data)
        for i, j in enumerate(self.label.tolist()):
            if j not in src_class_set:
                continue
            labels.append(j)
            # 对应class类型存入数据的index
            self.src_files[int(j)].append(i)

        for k, v in tgt_plabel_dict.items():
            if v in tgt_class_set:
                self.tgt_files[int(v)].append(k)

    def __getitem__(self, index):
        if self.num_steps is not None:
            index = index % (len(self.src_files))

        label = self.ind2label[index]
       
        src_pool = self.src_files[label]
    
        tgt_pool = self.tgt_files[label]
        
        src_index = np.random.choice(len(src_pool), self.num_pclass)
        tgt_index = np.random.choice(len(tgt_pool), self.num_pclass)

        src_path = [src_pool[i] for i in src_index]
        tgt_path = [tgt_pool[i] for i in tgt_index]
        src_labels = [label for i in src_index]
        tgt_labels = [label for i in tgt_index]
        src_labels = torch.Tensor(src_labels).long()
        tgt_labels = torch.Tensor(tgt_labels).long()
        src_imgs = self.src_data[src_path, :]
        tgt_imgs = self.target_data[tgt_path, :]
        if self.config.onehot:
            src_labels = one_hot_encoder(src_labels, self.config.num_classes)
        return src_imgs, tgt_imgs, src_labels

    def __len__(self):
        assert len(self.src_files) == len(self.tgt_files)
        if self.num_steps is None:
            return len(self.src_files)
        else:
            return self.num_steps
