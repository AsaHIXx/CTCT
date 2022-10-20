from model.ctc_model import Ctcnet
import torch
from collections import OrderedDict
from utils.vis import vis_transfor, vis_batch, vis_transfor_3
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
from utils.data_preprocessing import mkdir
from datetime import datetime
import torch
from sklearn.decomposition import PCA, SparsePCA, KernelPCA

torch.multiprocessing.set_sharing_strategy('file_system')
from init_config import *
from easydict import EasyDict as edict
import sys
import trainer
import copy
import numpy as np
import random
import importlib
from utils.postprocessing import return_class_acc
import matplotlib.pyplot as plt
import re
import pandas as pd
from utils.data_preprocessing import sampling_by_class, MatrixMaskByClassType, H5toCSV, get_new_label, tabular_read_in

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


def pred_process(exp, test_cell_t, model):
    where_are_inf = np.isinf(exp)
    exp[where_are_inf] = 0
    test_cell_t = np.asarray(test_cell_t)
    test_dataset = Data.TensorDataset(torch.from_numpy(exp).type(torch.FloatTensor),
                                      torch.from_numpy(test_cell_t).type(torch.FloatTensor))
    test_loader = Data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=280)
    correct = 0
    pred = torch.Tensor().long()
    conf = torch.Tensor()
    embed = torch.Tensor()
    layer_last = torch.Tensor()
    for i, (exps, labels) in enumerate(test_loader):
        embed_result, _, outputs, prob_labels, layer, _, _ = model(exps)
        _, preds_label = torch.max(prob_labels.data, 1)
        correct += (preds_label == labels).sum().item()
        conf = torch.cat((conf, prob_labels))
        pred = torch.cat((pred, preds_label))
        embed = torch.cat((embed, embed_result))
        layer_last = torch.cat((layer_last, layer))
    total_acc = correct / len(test_cell_t) * 100
    acc_dict = return_class_acc(pred, test_cell_t)
    print('ACC', acc_dict)
    print(total_acc, pred)
    return pred, acc_dict, total_acc, embed.detach().numpy(), layer_last.detach().numpy()

config, writer, message = init_config("config/vis.yaml", sys.argv)
start_time = datetime.now()
time_str = start_time.strftime('%Y_%m_%d_%H_%M')
path = config.snapshot + 'Visualization_' + time_str + '/'
mkdir(path)


def remove_z(string):
    new_str = ""
    for i in string:
        if i != '0':
            new_str = new_str + i
    return new_str


def re_sub(stri):
    if 'classifier.fc2' in stri:
        i = re.sub('.0', '', stri)
        return i
    else:
        return stri




#####################################
###############需要修改###############
#####################################
source_exp_data = tabular_read_in(config.source_data)
target_exp_data = tabular_read_in(config.target_data)
test_exp_data = tabular_read_in(config.test_data)
experiment_data = tabular_read_in(config.experiment_data)
label_for_source = tabular_read_in(config.source_label)
target_label_data = tabular_read_in(config.target_label)
test_label_data = tabular_read_in(config.test_label)
experiment_label = tabular_read_in(config.experiment_label)

# source_exp_data = np.load(config.source_data)
# source_exp_data = source_exp_data.astype(np.float32)
# target_exp_data = np.loadtxt(config.target_data, dtype=np.float32)
# label_for_source = np.loadtxt(config.source_label, dtype=np.float32)
# experiment_data = np.loadtxt(config.experiment_data, dtype=np.float32)
# experiment_label = np.loadtxt(config.experiment_label, dtype=np.float32)
# target_label_data = np.loadtxt(config.target_label, dtype=np.float32)
ensemble = './data_examples/gene_ensemble_18856.txt'
symbol = './data_examples/gene_hygo_18856.txt'

if config.source_addition != 'None':
    for idx in range(len(config.source_addition)):
        if config.source_addition[idx].split('.')[-1] == 'csv':
            exp_df = pd.read_csv(config.source_addition[idx], sep=',', index_col=0)
        elif config.source_addition[idx].split('.')[-1] == 'h5':
            h5_transer = H5toCSV(config.source_addition[idx], symbol)
            exp_df = h5_transer.select_gene()
            # exp_df = exp_df.sample(1500, axis=1, random_state=1234)
        elif config.source_addition[idx].split('.')[-1] == 'txt':
            exp_df = pd.read_csv(config.source_addition[idx], sep='\t', index_col=0)
        source_exp_data = np.append(source_exp_data, exp_df.values.T.astype(np.float32), axis=0)
        label_for_source = np.append(label_for_source, np.array(
            [int(config.source_addition_label[idx]) for i in range(exp_df.shape[1])]))
    print(np.array(
        [int(config.source_addition_label[idx]) for i in range(exp_df.shape[1])]))
    print('after ', config.source_addition[idx], 'addition, shape is', source_exp_data.shape,
          label_for_source.shape)
############ add data to target ##########
if config.target_addition != 'None':
    for idx in range(len(config.target_addition)):
        if config.target_addition[idx].split('.')[-1] == 'csv':
            exp_df = pd.read_csv(config.target_addition[idx], sep=',', index_col=0)
        elif config.target_addition[idx].split('.')[-1] == 'h5':
            h5_transer = H5toCSV(config.target_addition[idx], symbol)
            exp_df = h5_transer.select_gene()
            # exp_df = exp_df.sample(1500, axis=1, random_state=1234)
            exp_df = exp_df.iloc[:, 0:1000]
        elif config.target_addition[idx].split('.')[-1] == 'txt':
            exp_df = pd.read_csv(config.target_addition[idx], sep='\t', index_col=0)
        target_exp_data = np.append(target_exp_data, exp_df.values.T.astype(np.float32), axis=0)
        target_label_data = np.append(target_label_data, np.array(
            [config.target_addition_label[idx] for i in range(exp_df.shape[1])]))
    print('after ', config.target_addition[idx], 'addition, shape is', target_exp_data.shape,
          target_label_data.shape)

# test_exp_data = np.append(test_exp_data, experiment_data2, axis=0)
if config.target_sample != 'None':
    target_exp_data, target_label_data = sampling_by_class(target_exp_data,
                                                           target_label_data,
                                                           config.target_sample)

####source data sample

if config.source_sample != 'None':
    source_exp_data, label_for_source = sampling_by_class(source_exp_data,
                                                          label_for_source,
                                                          config.source_sample)

if config.mask != 'None':
    mmbt = MatrixMaskByClassType(target_exp_data, target_label_data, config.mask)
    target_exp_data, target_label_data = mmbt.mat_masked()
if config.source_mask != 'None':
    mmbt = MatrixMaskByClassType(source_exp_data, label_for_source, config.source_mask)
    source_exp_data, _ = mmbt.mat_masked()
    label_for_source = get_new_label(label_for_source, config.source_mask)
    target_label_data = get_new_label(target_label_data, config.source_mask)
    experiment_label = get_new_label(experiment_label, config.source_mask)

test_exp_data = np.append(target_exp_data, experiment_data, axis=0)
test_label = np.append(target_label_data, experiment_label)
X_eval, y_eval = source_exp_data, label_for_source
config.eval_class = len(set(label_for_source))
model = Ctcnet(num_inputs=config.inputs_dim, embed_size=config.embed_size, class_out=config.eval_class)
params = torch.load(config.check_model)
print('Model restored with weights from : {}'.format(config.check_model))
model.load_state_dict(params, strict=True)


border = len(y_eval)
border_ni = border + len(target_label_data)
X_eval = np.append(X_eval, test_exp_data, axis=0)
y_eval = np.append(y_eval, test_label)
np.savetxt(fname=path + 'split_source_label.txt', X=y_eval, fmt='%.0f')
REPORT = open(path + 'INFO_Result', 'a')
REPORT.write(str(model))
REPORT.write('\n')
REPORT.close()
pred_result, acc_dict, total_acc, embed_res, layer_res = pred_process(test_exp_data, test_label, model)
pred_result, acc_dict, total_acc, embed_res, layer_res = pred_process(X_eval, y_eval, model)
columns = ['Primary' for i in range(border)]
columns.extend(['CTC batch 1' for i in range(len(target_label_data))])
columns.extend(['CTC batch 2' for i in range(len(experiment_label))])

pca = PCA(n_components=10, random_state=1234)
X_eval_umap = X_eval
embed_res_umap = embed_res
layer_res_umap = layer_res
try:
    X_eval = pca.fit_transform(X_eval)
except:
    X_eval = pca.fit_transform(X_eval)
print('layer shape', layer_res.shape)
try:
    layer_res = pca.fit_transform(layer_res)
except:
    layer_res = pca.fit_transform(layer_res)
pred_result = pred_result.detach().numpy()

### save embedding matrix
cells_embedded_layer, cells_embedded_trans_layer = vis_transfor(X_eval, layer_res, y_eval, pred_result,
                                                                [i for i in range(config.eval_class)], border=border,
                                                                border_ni=border_ni,
                                                                tumor_name=config.tumor_name, path=path, infor='layer')
cells_embedded_layer2, cells_embedded_trans_layer2 = vis_transfor_3(X_eval, layer_res, y_eval, pred_result,
                                                                [i for i in range(config.eval_class)], border=border,
                                                                border_ni=border_ni,
                                                                tumor_name=config.tumor_name, path=path, infor='layer')