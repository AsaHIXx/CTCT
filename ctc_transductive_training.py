import torch

torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from dataset import *  # init_dataset
from model import *
from init_config import *
from easydict import EasyDict as edict
import sys
import trainer
import time, datetime
import copy
import numpy as np
import random
import importlib
import datetime
import os

cudnn.enabled = True
cudnn.benchmark = True

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path, 'make success!!')

    else:
        print('path already exists！！！')
    return None


start_time = datetime.datetime.now()
time_str = start_time.strftime('%Y_%m_%d_%H_%M')

config, writer, message = init_config("config/ctc_net.yaml", sys.argv)
Param = importlib.import_module('trainer.{}_trainer'.format(config.trainer))
os.environ["CUDA_VISIBLE_DEVICES"] = config.device
config.time_str = time_str
path_for_dir = config.snapshot
path = path_for_dir + config.cluster + '_' + config.train_info + time_str + '/'
mkdir(path)
config.snapshot = path
REPORT = open(path + 'Train_Result', 'a')
REPORT.write(message)
REPORT.write('\n'
             'Start_time:{}'
             '\n'.format(str(start_time)))
REPORT.close()
trainer = Param.Trainer(config, writer)
trainer.train()
end_time = datetime.datetime.now()



REPORT = open(path + 'Train_Result', 'a')
REPORT.write('Time consume:{}'
             '\n'.format(str(end_time - start_time)))
REPORT.close()
