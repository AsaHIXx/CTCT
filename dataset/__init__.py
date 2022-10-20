import numpy as np
import os.path as osp
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.utils.data as Data
from torch.utils import data
import torchvision.transforms.functional as TF
from torchvision.transforms.transforms import *
from .base_dataset import BaseDataset
from .class_dataset import ClassAwareDataset
import torch
import random
from .target_dataset import TargetClassAwareDataset
from utils.postprocessing import normalize_zscore
from collections import Counter


def get_transform(train=True):
    if train:
        transforms = Compose([
            Resize(256),
            RandomCrop(224),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transforms = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transforms


def get_path_dataset(config, path_dict, length=None, test=False, batch_size=None, get_loader=False):
    transforms = get_transform(train=not test)

    if length is not None and not test:
        num_steps = length * config.batch_size
    elif length is not None and test:
        num_steps = length
    else:
        num_steps = None

    dataset = PathDataset(path_dict, transforms, num_steps=num_steps)
    if batch_size is None:
        batch_size = config.batch_size
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=config.num_worker, shuffle=not test,
                        drop_last=not test)
    if get_loader:
        return loader
    else:
        return enumerate(loader)


def get_dataset(config, dataset, class_set, label_list=None, test=False, batch_size=None, plabel_dict=None,
                get_loader=True, length=None, binary_label=None, class_wise=False, validate=False):
    if length is not None and not test:
        num_steps = length * config.batch_size
    elif length is not None and test:
        num_steps = length
    else:
        num_steps = None
    if class_wise:
        TargetClassAwareDataset(config.num_pclass, class_set, plabel_dict,
                                num_steps=length * config.num_sample)
    else:
        dataset = BaseDataset(dataset, class_set, num_steps=num_steps,
                              plabel_dict=plabel_dict, binary_label=binary_label)
    if batch_size is None:
        batch_size = config.batch_size

    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=config.num_worker, shuffle=True,
                        drop_last=not test)
    if get_loader:
        return loader
    else:
        return dataset

def weight_sampler(labels, balance=True):
    
    classes = labels
    freq = Counter(classes)
    class_weight = {x: 1.0 / freq[x] if balance else 1.0 for x in freq}
    source_weights = [class_weight[x] for x in labels]
    sampler = WeightedRandomSampler(source_weights, len(labels))
    return sampler


def _get_dataset(config, exp, label, sampler=False):
    dataset = BaseDataset(exp, label, config)
    if sampler:
        data_loader = Data.DataLoader(dataset=dataset, batch_size=config.batch_size, sampler=sampler)
    else:
        data_loader = Data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=True)
    return data_loader


def get_dataset_(source, source_class, batch_size, sampler=None):
    print('source class shape is ', source.shape, source_class.shape)
    dataset = Data.TensorDataset(torch.from_numpy(source), torch.from_numpy(source_class))
    if sampler is None:
        data_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    else:
        data_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)
    return data_loader


def init_pair_dataset(config, source_data, source_label, target_data, target_label, src_wei=None, label_list=None,
                      plabel_dict=None, length=None, binary_label=None):
    
    sampler = weight_sampler(target_label)
    src_loader = get_dataset_(source_data, source_label, config.batch_size)
    tgt_loader = get_dataset_(target_data, target_label, config.batch_size, sampler=None)
    src_loader = enumerate(src_loader)
    tgt_loader = enumerate(tgt_loader)

    return src_loader, tgt_loader  # , s_test_loader, t_test_loader


def init_class_dataset(config, source_data, target_data, source_label, plabel_dict, src_class_set, tgt_class_set,
                       length=None, uk_list=None):

    dataset = ClassAwareDataset(config, source_data, target_data, source_label, config.num_pclass, src_class_set,
                                tgt_class_set,
                                plabel_dict, num_steps=length * config.num_sample, uk_list=uk_list)
    dataloader = DataLoader(dataset=dataset, batch_size=config.num_sample, num_workers=config.num_worker, shuffle=True,
                            drop_last=False)
    dataloader = enumerate(dataloader)
    return dataloader


def init_target_dataset(config, target_data, plabel_dict, tgt_class_set, length=None, uk_list=None, binary_label=None):

    dataset = TargetClassAwareDataset(config, target_data, config.num_pclass, tgt_class_set, plabel_dict,
                                      num_steps=length * config.num_sample)
    dataloader = DataLoader(dataset=dataset, batch_size=config.num_sample, num_workers=config.num_worker, shuffle=True,
                            drop_last=False)
    dataloader = enumerate(dataloader)
    return dataloader
