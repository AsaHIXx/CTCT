from operator import index
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import torch
import torch.utils.data as Data
from datetime import datetime
from model.ctc_model import Ctcnet
import yaml
from easydict import EasyDict as edict
from collections import OrderedDict, Counter
import re
import os
import argparse


parser = argparse.ArgumentParser(description='Prediction')
parser.add_argument('-i','--input_file_path', type=str, default='data_examples/nature_306_brca_logtpm.csv')
parser.add_argument('-r','--ref_label', type=str, default='None')
parser.add_argument('-t','--tumor_info', type=str, default='data_examples/tumor_ident_num.txt')
parser.add_argument('-c','--class_out', type=int, default=26)
parser.add_argument('-m', '--model', type=str, default='pretrained_model/CTC_pretrained_model.pth')
args = parser.parse_args()


start_time = datetime.now()
time_str = start_time.strftime('%Y_%m_%d_%H_%M')


def net_loader(config):
    model = Ctcnet(num_inputs=config.input_dim, embed_size=config.embed_size, class_out=config.class_out)
    params = torch.load(config.pretrained_model, map_location=torch.device('cpu'))
    model.load_state_dict(params, strict=True)
    return model
def pred_process(exp, model, config):
    where_are_inf = np.isinf(exp)
    exp[where_are_inf] = 0
    test_cell_t = np.array([5 for i in range(exp.shape[0])])
    test_dataset = Data.TensorDataset(torch.from_numpy(exp).type(torch.FloatTensor),
                                      torch.from_numpy(test_cell_t).type(torch.FloatTensor))
    test_loader = Data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=config.batch_size)
    correct = 0
    pred = []
    conf = torch.Tensor()
    max_conf = []
    for i, (exps, labels) in enumerate(test_loader):
        pred_tmp = []
        max_conf_tmp = []
        _, neck, outputs, prob_labels, _, _, _ = model(exps)
        # _, preds_label = torch.max(prob_labels, 1)
        for i in prob_labels:
            if torch.max(i,0) == torch.min(i,0):
                _, preds_label = torch.max(i, 0)
                max_conf_tmp.append(_.detach())
                pred_tmp.append('Null')
            else:
                _, preds_label = torch.max(i, 0)
                max_conf_tmp.append(_.detach())
                pred_tmp.append(preds_label.detach())
        # print(torch.max(prob_labels,1))
        max_conf.extend(max_conf_tmp)
        
        conf = torch.cat((conf, prob_labels))
        pred.extend(pred_tmp)
    return conf, pred, max_conf

def easy_dic(dic):
    dic = edict(dic)
    for key, value in dic.items():
        if isinstance(value, dict):
            dic[key] = edict(value)
    return dic

def tabular_read_in(path):
    if path.split('.')[-1] == 'npy':
        
        tabular = np.load(path)
        tabular = tabular.astype(np.float32)
    elif path.split('.')[-1] == 'csv':
        tabular = pd.read_csv(path, dtype=np.float32,index_col=0)
        tabular = tabular.values.T.astype(np.float32)
        
    elif path.split('.')[-1] == 'txt':
        tabular = np.loadtxt(path, dtype=np.float32)
    print(path,'read in', 'data shape is ', tabular.shape)
    return tabular

def read_list(path):
    a = pd.read_csv(path, sep='\s+', names=['index', 'names'])
    lis = list(a['names'])
    return lis

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path, 'make success!!')

    else:
        print('path already exists！！！')
    return None
def isDigit(str):
 
    try: 
        f = float(str) 
    except ValueError: 
        return False
    else:
        return True

if __name__ == '__main__':
    start_time = datetime.now()
    time_str = start_time.strftime('%Y_%m_%d_%H_%M')
    config_path = './config/val.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    config = easy_dic(config)
    # config.input_data = args.input_file_path
    # config.tumor_info = args.tumor_info
    # config.ref_label = args.ref_label
    # config.class_out = args.class_out
    # config.pretrained_model = args.model
    path = config.log + time_str + '/'
    mkdir(path)
    input_file_path = config.input_data
    tumor_info = read_list(config.tumor_info)
    if input_file_path.split('.')[-1] in ['csv']:
        X = pd.read_csv(input_file_path, index_col=0)
        cell_name = list(X.columns)
        X = X.values.T.astype(np.float32)
    else:
        X = tabular_read_in(input_file_path)
        cell_name = ['cell_'+str(i+1) for i in range(X.shape[0])]
    if config.ref_label != 'None':
        tumor_ref = tabular_read_in(config.ref_label)
        tumor_ref_type = [tumor_info[int(i)] for i in tumor_ref]
    else:
        tumor_ref_type = ['None'] * X.shape[0]
        
    
    model = net_loader(config)
    model.eval()
    conf, pred, max_conf = pred_process(X, model, config)
    tumor_pred = [] 
    for i in pred:
        if isDigit(i):
            tumor_pred.append(tumor_info[i])
        else:
            tumor_pred.append(i)
    with open(path+'PredResults', 'w') as f:
        f.write('cell_name'
                ','
                'pred'
                ','
                'softmax_value'
                ','
                'ref_type'
                '\n')
        for i in range(len(cell_name)):
            f.write('{}'
                    ','
                    '{}'
                    ','
                    '{}'
                    ','
                    '{}'
                    '\n'.format(cell_name[i], tumor_pred[i], max_conf[i], tumor_ref_type[i])
                
            )
        f.close()
    pred = [int(i) for i in pred]
    if config.ref_label != 'None':
        
        with open(path+'PredResults', 'a') as f:
            f.write('pred_acc:{}'.format(np.mean(tumor_ref == pred)))
            f.close()
        print('######################################################\n')
        print(input_file_path, ' total acc:', np.mean(tumor_ref == pred))
    if config.class_out == 2:
        print('total number of tumor cells is:', Counter(pred)[0])
        
    end_time = datetime.now()
    time_consume = end_time-start_time
    print('Done! Time consume is:', time_consume)  
     


