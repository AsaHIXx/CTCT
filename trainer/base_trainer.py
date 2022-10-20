import torch
import os.path as osp
import torch.nn as nn
# import neptune.new as neptune
from tqdm import tqdm
import operator
import math
import torch.optim as optim
from utils.optimize import *
from easydict import EasyDict as edict
from utils import *
from utils.memory import *
from utils.flatwhite import *
from dataset import *
from sklearn import metrics
import sklearn
from sklearn.cluster import KMeans
# from sklearn.cluster import SpectralClustering as KMeans
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as Data
from dataset import _get_dataset
from utils.postprocessing import return_class_acc
import pandas as pd
from imblearn.over_sampling import SMOTE

from collections import Counter
from utils.postprocessing import normalize_zscore, find_duplciates1
from utils.data_preprocessing import gene_feature_selection, MatrixMaskByClassType, sampling_by_class, H5toCSV, \
    get_new_label, sample_random_split, tabular_read_in
from utils.feature_selection import Dimension_reduce, VarianceMeanFeatureSelection, MultiCountsFileReshape
from utils.postprocessing import one_hot_encoder


def get_dataset_Test(config, source, source_class, batch_size, shuffle=False):
    dataset = Data.TensorDataset(torch.from_numpy(source), torch.from_numpy(source_class))
    data_loader = Data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle)
    return data_loader


class BaseTrainer(object):
    def __init__(self, config, writer):

        self.config = config
        self.writer = writer

        self.flag = 0
        self.epoch_num = 0
        self.s_loss_epoch = 0
        self.cdd_loss_epoch = 0
        self.t_loss_epoch = 0
        self.total_loss_epoch = 0
        self.iter_num = 0
        self.num_center_counts = 0
        self.acc_list = [0]*26
        self.gene_selected = '/data/home/scv4524/run/xx/ANN/dataset/gene_selected_1714.txt'
        if self.config.tensorboard:
            self.writer = SummaryWriter(osp.join(self.config.snapshot, 'log'))
        self.best = 0.0

        if self.config.neuron_add != 'None':
            # self.config.class_out += self.config.add_neuron
            self.config.cls_share += int(self.config.neuron_add)
            # self.config.num_classes += self.config.add_neuron
        self.ensemble = '/disk/sdb/gxx/task/CTC/feature_folder/gene_ensemble_18856.txt'
        self.symbol = '/disk/sdb/gxx/task/CTC/feature_folder/gene_hygo_18856.txt'
        self.acc_best_h = 0.0
        self.h_best_acc = 0.0
        self.k_best = 0.0
        self.h_best = 0.0
        self.label_mask = None
        self.k_converge = False
        self.score_vec = None
        self.test_score = 0
        self.clus_acc_num = 0
        self.source_center = np.random.randn(1, 200)
        self.target_center = np.random.randn(1, 200)
        self.center_index = []
        self.target_index = []
        self.acc_dict_for_test = {}
        #######################
        ########数据读入########
        #######################
        # if self.config.normalize:
        #     self.source_exp_data = normalize_zscore(np.loadtxt(self.config.source_data, dtype=np.float32))
        #     self.target_exp_data = normalize_zscore(np.loadtxt(self.config.target_data, dtype=np.float32))
        #     self.test_exp_data = normalize_zscore(np.loadtxt(self.config.test_data, dtype=np.float32))
        #     self.experiment_data = normalize_zscore(np.loadtxt(self.config.experiment_data, dtype=np.float32))
        # else:
        #     if self.config.source_data.split('.')[-1] == 'npy':
        #         self.source_exp_data = np.load(self.config.source_data)
        #         self.source_exp_data = self.source_exp_data.astype(np.float32)
        #     else:
        #         self.source_exp_data = np.loadtxt(self.config.source_data, dtype=np.float32)
        #     if self.config.target_data.split('.')[-1] == 'npy':
        #         self.target_exp_data = np.load(self.config.target_data)
        #         self.target_exp_data = self.target_exp_data.astype(np.float32)
        #     else:
        #         self.target_exp_data = np.loadtxt(self.config.target_data, dtype=np.float32)
        #     self.test_exp_data = np.loadtxt(self.config.test_data, dtype=np.float32)
        #     self.experiment_data = np.loadtxt(self.config.experiment_data, dtype=np.float32)
        #     # self.experiment_data2 = np.loadtxt(self.config.experiment_data2, dtype=np.float32)
        
        # self.label_for_source = np.loadtxt(self.config.source_label, dtype=np.float32)
        # self.target_label_data = np.loadtxt(self.config.target_label, dtype=np.float32)
        # self.test_label_data = np.loadtxt(self.config.test_label, dtype=np.float32)
        # self.experiment_label = np.loadtxt(self.config.experiment_label, dtype=np.int16)
        self.source_exp_data = tabular_read_in(self.config.source_data)
        self.target_exp_data = tabular_read_in(self.config.target_data)
        self.test_exp_data = tabular_read_in(self.config.test_data)
        self.experiment_data = tabular_read_in(self.config.experiment_data)
        self.label_for_source = tabular_read_in(self.config.source_label)
        self.target_label_data = tabular_read_in(self.config.target_label)
        self.test_label_data = tabular_read_in(self.config.test_label)
        self.experiment_label = tabular_read_in(self.config.experiment_label)
        if self.config.balance_smote != 'None':
            smo = SMOTE(random_state=1234)
            self.source_exp_data, self.label_for_source = smo.fit_resample(self.source_exp_data, self.label_for_source)
            self.target_exp_data, self.target_label_data = smo.fit_resample(self.target_exp_data, self.target_label_data)
        if self.config.all_sample != 'None':
            self.target_exp_data, self.target_label_data = sample_random_split(self.target_exp_data,
                                                                               self.target_label_data,
                                                                               self.config.all_sample,
                                                                               self.config.seeds_change)
        ###############
        ########给source塞数据
        ###############
        if self.config.source_addition != 'None':
            for idx in range(len(self.config.source_addition)):
                if self.config.source_addition[idx].split('.')[-1] == 'csv':
                    exp_df = pd.read_csv(self.config.source_addition[idx], sep=',', index_col=0)
                elif self.config.source_addition[idx].split('.')[-1] == 'h5':
                    h5_transer = H5toCSV(self.config.source_addition[idx], self.symbol)
                    exp_df = h5_transer.select_gene()
                    # exp_df = exp_df.sample(1500, axis=1, random_state=1234)
                elif self.config.source_addition[idx].split('.')[-1] == 'txt':
                    exp_df = pd.read_csv(self.config.source_addition[idx], sep='\t', index_col=0)
                self.source_exp_data = np.append(self.source_exp_data, exp_df.values.T.astype(np.float32), axis=0)
                self.label_for_source = np.append(self.label_for_source, np.array(
                    [int(self.config.source_addition_label[idx]) for i in range(exp_df.shape[1])]))
                print(np.array(
                    [int(self.config.source_addition_label[idx]) for i in range(exp_df.shape[1])]))
                print('after ', self.config.source_addition[idx], 'addition, shape is', self.source_exp_data.shape,
                      self.label_for_source.shape)
        ############ add data to target ##########
        if self.config.target_addition != 'None':
            for idx in range(len(self.config.target_addition)):
                if self.config.target_addition[idx].split('.')[-1] == 'csv':
                    exp_df = pd.read_csv(self.config.target_addition[idx], sep=',', index_col=0)
                    exp_df = exp_df.values.T
                elif self.config.target_addition[idx].split('.')[-1] == 'h5':
                    h5_transer = H5toCSV(self.config.target_addition[idx], self.symbol)
                    exp_df = h5_transer.select_gene()
                    # exp_df = exp_df.sample(1500, axis=1, random_state=1234)
                    # exp_df = exp_df.iloc[:, 0:1000]
                    exp_df = exp_df.values.T
                elif self.config.target_addition[idx].split('.')[-1] == 'txt':
                    try:
                        exp_df = pd.read_csv(self.config.target_addition[idx], sep='\t', index_col=0)
                        exp_df = exp_df.values.T
                    except:
                        exp_df = np.loadtxt(self.config.target_addition[idx])
                self.target_exp_data = np.append(self.target_exp_data, exp_df, axis=0)
                self.target_label_data = np.append(self.target_label_data, np.array(
                    [self.config.target_addition_label[idx] for i in range(exp_df.shape[0])]))
                print('after ', self.config.target_addition[idx], 'addition, shape is', self.target_exp_data.shape,
                      self.target_label_data.shape)
        ########################
        ########增加experiment数据增加
        if self.config.experiment_addition != 'None':
            for idx in range(len(self.config.experiment_addition)):
                if self.config.experiment_addition[idx].split('.')[-1] == 'csv':
                    exp_df = pd.read_csv(self.config.experiment_addition[idx], sep=',', index_col=0)
                    exp_df = exp_df.values.T
                elif self.config.experiment_addition[idx].split('.')[-1] == 'h5':
                    h5_transer = H5toCSV(self.config.experiment_addition[idx], self.symbol)
                    exp_df = h5_transer.select_gene()
                    # exp_df = exp_df.sample(1500, axis=1, random_state=1234)
                    # exp_df = exp_df.iloc[:, 0:1000]
                    exp_df = exp_df.values.T
                elif self.config.experiment_addition[idx].split('.')[-1] == 'txt':
                    try:
                        exp_df = pd.read_csv(self.config.experiment_addition[idx], sep='\t', index_col=0)
                        exp_df = exp_df.values.T
                    except:
                        exp_df = np.loadtxt(self.config.experiment_addition[idx])
                self.experiment_data = np.append(self.experiment_data, exp_df, axis=0)
                self.experiment_label = np.append(self.experiment_label, np.array(
                    [self.config.experiment_addition_label[idx] for i in range(exp_df.shape[0])]))
                print('after ', self.config.experiment_addition[idx], 'addition, shape is', self.experiment_data.shape,
                      self.experiment_label.shape)

        #########################
        ######添加额外数据#########
        #########################
        if self.config.addition_source != 'None':
            self.addit_source = np.loadtxt(self.config.addition_source, dtype=np.float32)
            self.source_exp_data = np.append(self.source_exp_data, self.addit_source, axis=0)
            self.addit_label = np.array([26 for i in range(self.addit_source.shape[0])])
            self.label_for_source = np.append(self.label_for_source, self.addit_label)

        if self.config.mask != 'None':
            mmbt = MatrixMaskByClassType(self.target_exp_data, self.target_label_data, self.config.mask)
            self.target_exp_data, self.target_label_data = mmbt.mat_masked()
            mmbt_2 = MatrixMaskByClassType(self.test_exp_data, self.test_label_data, self.config.mask)
            self.test_exp_data, self.test_label_data = mmbt_2.mat_masked()
        if self.config.source_mask != 'None':
            mmbt = MatrixMaskByClassType(self.source_exp_data, self.label_for_source, self.config.source_mask)
            self.source_exp_data, _ = mmbt.mat_masked()
            self.label_for_source = get_new_label(self.label_for_source, self.config.source_mask)
            self.target_label_data = get_new_label(self.target_label_data, self.config.source_mask)
            self.test_label_data = get_new_label(self.test_label_data, self.config.source_mask)
            self.experiment_label = get_new_label(self.experiment_label, self.config.source_mask)

        self.config.class_out = len(set(self.label_for_source))
        self.config.num_classes = len(set(self.label_for_source))
        # if self.config.target_addition != 'None':

        ########## target 数据 sample
        if self.config.target_sample != 'None':
            self.target_exp_data, self.target_label_data = sampling_by_class(self.target_exp_data,
                                                                             self.target_label_data,
                                                                             self.config.target_sample,
                                                                             self.config.target_sample_class,
                                                                             self.config.seeds_change)
        ####source data sample
        if self.config.source_sample != 'None':
            self.source_exp_data, self.label_for_source = sampling_by_class(self.source_exp_data,
                                                                            self.label_for_source,
                                                                            self.config.source_sample,
                                                                            list(set(self.label_for_source)))
        if self.config.testequaltarget != 'None':
            self.test_label_data = self.target_label_data
            self.test_exp_data = self.target_exp_data
        self.test_exp_data = np.append(self.test_exp_data, self.experiment_data, axis=0)
        self.test_label_data = np.append(self.test_label_data, self.experiment_label)
        ## target 数据特征筛选
        if self.config.feature_selected != 'None':
            vmfs = VarianceMeanFeatureSelection(
                path=self.config.experimentpath,
                variance_threshold=10,
                mean_threshold=5,
                file_short='/*log.csv')
            self.gene_experiment, _ = vmfs.feature_selection()
        ##################################
        ########source/target数据维度降低###
        ##################################
        if self.config.variance != 'None':
            Dimension_reducer = Dimension_reduce(source_arr=self.source_exp_data, target_arr=self.target_exp_data,
                                                 variance=self.config.variance, mean=5, save_path=self.config.snapshot)
            self.target_exp_data, self.source_exp_data, self.config.inputs_dim, self.gene_selected = Dimension_reducer.feature_selection()
            self.test_exp_data = gene_feature_selection(self.test_exp_data, self.gene_selected)
            self.experiment_data = gene_feature_selection(self.experiment_data, self.gene_selected)
        self.exp_result = self.experiment_data.shape[0]
        ###### 输出控制
        self.config.source_sample_num = self.source_exp_data.shape[0]
        self.config.target_sample_num = self.target_exp_data.shape[0]
        self.epoch_iter = int(self.config.target_sample_num / self.config.batch_size) + 1
        self.config.print_freq = self.epoch_iter
        self.config.val_freq = self.epoch_iter
        self.print_freq_epoch = 50
        self.config.save_freq = self.print_freq_epoch * 20
        #########记录log
        target_train_info, _ = find_duplciates1(self.target_label_data)

        REPORT = open(self.config.snapshot + 'Train_Result', 'a')
        REPORT.write('train source shape, and train target shape is {}, {}'.format(self.source_exp_data.shape,
                                                                                   self.target_exp_data.shape))
        REPORT.write('\n')

        REPORT.write('train target info {}'.format(str(target_train_info)))

        REPORT.write('\n')
        REPORT.close()
        ##################################
        ##########样本平衡采样##############
        ##################################
        classes = self.label_for_source
        freq = Counter(classes)
        class_weight = {x: 1.0 / freq[x] if self.config.class_balance else 1.0 for x in freq}
        source_weights = [class_weight[x] for x in self.label_for_source]
        sampler = WeightedRandomSampler(source_weights, len(self.label_for_source))
        self.sampler = sampler
        # if not self.config.class_balance:
        #     self.sampler = False
            
        #################################
        #########数据打包##################
        ##################################
        self.test_loader = get_dataset_Test(config, self.test_exp_data,
                                            self.test_label_data,
                                            batch_size=64)
        self.tgt_loader = _get_dataset(config, self.target_exp_data, self.target_label_data)
        self.src_loader = _get_dataset(config, self.source_exp_data, self.label_for_source, sampler=self.sampler)

        self.best_prec = 0.0
        self.best_recall = 0.0

    def train(self):
        # print('train_step__base')
        for i_iter in range(self.config.num_steps):
            losses = self.iter(i_iter)
            if i_iter % self.config.print_freq == 0:
                self.print_loss(i_iter)
                self.save_model(i_iter)
                print('model for iter_{} saved !! '.format(i_iter))
            if i_iter % self.config.save_freq == 0:
                self.save_model(i_iter)
                print('model for iter_{} saved !! '.format(i_iter))
            if self.config.val and i_iter % self.config.val_freq == 0 and i_iter != 0:
                self.validate()

    def save_model(self, iter, info: str = 'General Train'):
        self.model.eval()
        tmp_name = str(iter) + '_' + info + '.pth'
        torch.save(self.model.state_dict(), self.config.snapshot + tmp_name)
        # self.model.train()

    def save_txt(self):
        with open(osp.join(self.config.snapshot, 'Train_Result'), 'a') as f:
            f.write('NormalCancer' + '->' + 'CTC_tumor_cell' + '[best]: ' + str(self.best) + ' ' + str(
                self.k_best) + ' [H-Score]: ' + str(self.h_best) + ' ' + str(self.acc_best_h) + ' ' + str(
                self.h_best_acc) + ' ' + str(self.best_prec) + ' ' + str(self.best_recall) + '\n')
            f.write('NormalCancer' + '->' + 'CTC_tumor_cell' + '[last]: ' + str(self.last) + ' ' + str(
                self.k_last) + ' [H-Score]: ' + str(self.h_last) + ' ' + str(self.last_prec) + ' ' + str(
                self.last_recall) + '\n')
            f.close()

    def print_loss(self, iter):
        iter_infor = ('iter = {:6d}/{:6d}, exp = {}, lr = {}'.format(iter, self.config.num_steps, self.config.note,
                                                                     self.optimizer.param_groups[0]['lr']))
        to_print = ['{}:{:.4f}'.format(key, self.losses[key].item()) for key in self.losses.keys()]
        loss_infor = '  '.join(to_print)
        if self.warmup and self.config.screen:
            print(iter_infor)
        if self.config.screen:
            print(iter_infor + '  ' + loss_infor)
            REPORT = open(self.config.snapshot + 'Train_Result', 'a')
            REPORT.write(iter_infor + '  ' + loss_infor)
            REPORT.write('\n')
            REPORT.close()
        if self.config.neptune:
            for key in self.losses.keys():
                self.neptune_metric('train/' + key, self.losses[key].item(), False)
        if self.config.tensorboard and self.writer is not None:
            for key in self.losses.keys():
                self.writer.add_scalar('train/' + key, self.losses[key], iter)

    def print_acc(self, acc_dict):
        str_dict = [str(k) + ': {:.2f}'.format(v) for k, v in acc_dict.items()]
        output = ' '.join(str_dict)
        print(output)

    def cos_simi(self, x1, x2):
        simi = torch.matmul(x1, x2.transpose(0, 1))
        return simi

    def gather_feats(self):
        data_feat, data_gt, data_paths, data_probs = [], [], [], []
        gts = []
        gt = {}
        preds = []
        names = []
        for _, batch in tqdm(enumerate(self.tgt_loader)):
            img, label, name = batch
            names.extend(name.numpy().tolist())
            img = img.to(torch.float32)
            with torch.no_grad():
                _, output, _, prob, _, _, _ = self.model(img.cuda())
            feature = output  # view(1,-1)
            N, C = feature.shape
            data_feat.extend(torch.chunk(feature, N, dim=0))
            gts.extend(torch.chunk(label, N, dim=0))

        for k, v in zip(names, gts):
            gt[k] = v.cuda()

        feats = torch.cat(data_feat, dim=0)
        feats = F.normalize(feats, p=2, dim=-1)
        print('names:', names)
        print('gt:', gt)
        return feats, gt, preds

    def validate(self, i_iter, class_set):
        self.model.eval()
        print('global label_set', self.global_label_set)
        if not self.config.prior:
            if self.config.num_centers == len(self.cluster_mapping):
                result = self.close_validate(i_iter)
            else:
                result = self.open_validate(i_iter)
        elif self.config.setting in ['uda', 'osda']:
            result = self.open_validate(i_iter)
        else:
            result = self.close_validate(i_iter)
        # result = self.open_validate(i_iter)
        over_all, k, h_score, recall, precision = result

        if over_all > self.best:
            self.best = over_all
            self.k_best = k
            self.acc_best_h = h_score
        if h_score > self.h_best:
            self.h_best = h_score
            self.h_best_acc = over_all
            self.best_recall = recall
            self.best_prec = precision
        if i_iter + 1 == self.config.stop_steps:
            self.last = over_all
            self.k_last = k
            self.h_last = h_score
            self.last_recall = recall
            self.last_prec = precision

        return result

    def close_validate(self, i_iter):
        self.model.train(False)
        self.model.eval()
        knows = 0.0
        unknows = 0.0
        k_co = 0.0
        uk_co = 0.0
        accs = GroupAverageMeter()
        # test_loader = get_dataset(self.config, self.config.target, self.config.target_classes, batch_size=100,
        #                           test=True)
        common_index = torch.Tensor(self.global_label_set).cuda().long()
        pred_result = []
        for _, batch in tqdm(enumerate(self.test_loader)):
            acc_dict = {}
            img, label = batch
            label = label.cuda()
            img = img.cuda()
            img = img.to(torch.float32)
            with torch.no_grad():
                _, neck, pred, pred2, _, _, _ = self.model(img)

            pred_label = pred2.argmax(dim=-1)
            pred_result.append(pred_label)
            label = torch.where(label >= self.config.num_classes, torch.Tensor([self.config.num_classes]).cuda(),
                                label.float())
            for i in label.unique().tolist():
                mask = label == i
                count = mask.sum().float()
                correct = (pred_label == label) * mask
                correct = correct.sum().float()

                acc_dict[i] = ((correct / count).item(), count.item())
            accs.update(acc_dict)
        acc = np.mean(list(accs.avg.values()))
        print('pred_result:', pred_result)
        print('acc:', acc)
        ground_truth = self.test_label_data
        pred_result = torch.cat(pred_result)
        pred_result = pred_result.cpu().numpy().tolist()
        pred_acc = np.mean(np.array(pred_result[0:-self.exp_result]) == ground_truth[0:-self.exp_result])

        self.acc_result = pred_acc
        acc_dict = return_class_acc(np.array(pred_result)[0:-self.exp_result, ], ground_truth[0:-self.exp_result, ])
        pred_acc_for_test = np.mean(np.array(pred_result)[-self.exp_result:, ] == ground_truth[-self.exp_result:, ])
        acc_dict_for_test = return_class_acc(np.array(pred_result)[-self.exp_result:, ],
                                             ground_truth[-self.exp_result:, ])
        self.acc_test_result = pred_acc_for_test
        self.acc_dict = acc_dict
        self.acc_dict_for_test = acc_dict_for_test
        print('acc_dict_for_test:', acc_dict_for_test)

        if (i_iter + 1) % self.print_freq_epoch == 0:
            REPORT = open(self.config.snapshot + 'Train_Result', 'a')
            REPORT.write('############Total############\n')
            REPORT.write(str(np.array(pred_result)[0:-self.exp_result, ]))
            REPORT.write('\n')
            REPORT.write('Total ACC:{}'.format(pred_acc))
            REPORT.write('\n')
            REPORT.write('Each Class ACC:')
            REPORT.write(str(acc_dict))
            REPORT.write('\n')
            REPORT.write('##############TEST###########\n')
            REPORT.write(str(np.array(pred_result)[-self.exp_result:, ]))
            REPORT.write('\n')
            REPORT.write('Total ACC:{}'.format(pred_acc_for_test))
            REPORT.write('\n')
            REPORT.write('Each Class ACC:')
            REPORT.write(str(acc_dict_for_test))
            REPORT.write('\n')
            REPORT.close()
        self.writer.add_scalar('val/test_acc' + str(i), pred_acc_for_test, self.iter_num)
        # 写入训练结果
        for i in list(acc_dict.keys()):
            self.writer.add_scalar('val/class' + str(i), acc_dict[i], self.iter_num)
            # self.acc_list[int(i)] += acc_dict[i]
        # if i_iter % self.epoch_iter == 0 and i_iter != 0:
        #     for i in list(acc_dict.keys()):
        #         self.writer.add_scalar('val/class_epoch' + str(i), self.acc_list[int(i)]/self.epoch_iter, self.epoch_num)
            
            self.acc_list = [0] * 26
        self.print_acc(accs.avg)
        if acc > self.best:
            self.best = acc
        self.model.train(True)
        # self.neptune_metric('val/Test Accuracy', acc)
        return acc, 0.0, 0.0, 0.0, 0.0

    def open_validate(self, i_iter):
        # self.model.set_bn_domain(1)
        self.model.train(False)
        self.model.eval()
        knows = 0.0
        unknows = 0.0
        accs = GroupAverageMeter()
        t_centers = self.memory.memory

        length = len(self.test_loader.sampler)
        cls_pred_all = torch.zeros(length).cuda()
        memo_pred_all = torch.zeros(length).cuda()
        gt_all = torch.zeros(length).cuda()
        uk_index = self.config.num_classes

        cnt = 0
        pred_result = []
        max_conf = []
        # pred_result = torch.Tensor()
        for _, batch in tqdm(enumerate(self.test_loader)):
            acc_dict = {}
            img, label = batch
            label = label.cuda()

            img = img.cuda()
            img = img.to(torch.float32)
            with torch.no_grad():
                _, neck, pred, pred2, _, _, _ = self.model(img)
            N = neck.shape[0]
            simi2cluster = self.cos_simi(F.normalize(neck, p=2, dim=-1), t_centers)
            clus_index = simi2cluster.argmax(dim=-1)
            cls_pred = pred2.argmax(-1)
            pred_result.append(cls_pred)
            # pred_result = torch.cat(pred_result, cls_pred)
            cls_pred_all[cnt:cnt + N] = cls_pred.squeeze()
            memo_pred_all[cnt:cnt + N] = clus_index.squeeze()
            gt_all[cnt:cnt + N] = label.squeeze()
            cnt += N
            counts=0
            max_conf_tmp = []
            if self.config.softmax_coverge:
                for j in label:
                    j = int(j)
                    max_conf_tmp.append(float(format(pred2[counts, j].detach().cpu().numpy(), '.6f')))
                    counts+=1
                max_conf.extend(max_conf_tmp)
        # print(torch.max(prob_labels,1))
        # _, preds_label = torch.max(i, 0)
        # pred_tmp.append(preds_label.detach())
        # pred.extend(pred_tmp)
            
        print('pred_result:', pred_result)
        pred_result = torch.cat(pred_result)
        pred_result = pred_result.cpu().numpy().tolist()
        # pred_result = sum(pred_result, [])
        ground_truth = self.test_label_data
        pred_acc = np.mean(np.array(np.array(pred_result)[0:-self.exp_result, ]) == ground_truth[0:-self.exp_result, ])
        self.acc_result = pred_acc
        acc_dict = return_class_acc(np.array(pred_result)[0:-self.exp_result, ], ground_truth[0:-self.exp_result, ])
        self.acc_dict = acc_dict
        
        print('pred_acc:', pred_acc)
        print('class_acc:', acc_dict)
        pred_acc_for_test = np.mean(np.array(pred_result)[-self.exp_result:, ] == ground_truth[-self.exp_result:, ])
        acc_dict_for_test = return_class_acc(np.array(pred_result)[-self.exp_result:, ],
                                             ground_truth[-self.exp_result:, ])
        self.acc_test_result = pred_acc_for_test
        self.acc_dict_for_test = acc_dict_for_test
        print('acc_dict_for_test:', acc_dict_for_test)
        if (i_iter + 1) % self.print_freq_epoch == 0:
            REPORT = open(self.config.snapshot + 'Train_Result', 'a')
            REPORT.write('############Total############\n')
            REPORT.write(str(np.array(pred_result)[0:-self.exp_result, ]))
            REPORT.write('\n')
            REPORT.write('Total ACC:{}'.format(pred_acc))
            REPORT.write('\n')
            REPORT.write('Each Class ACC:')
            REPORT.write(str(acc_dict))
            REPORT.write('\n')
            REPORT.write('##############TEST###########\n')
            REPORT.write(str(np.array(pred_result)[-self.exp_result:, ]))
            REPORT.write('\n')
            REPORT.write('Total ACC:{}'.format(pred_acc_for_test))
            REPORT.write('\n')
            REPORT.write('Each Class ACC:')
            REPORT.write(str(acc_dict_for_test))
            REPORT.write('\n')
            REPORT.close()
        self.writer.add_scalar('val/test_acc', pred_acc_for_test, self.iter_num)
        if self.config.softmax_coverge:
            self.writer.add_scalar('softmax_value_for_experiment', np.mean(max_conf[-self.exp_result:]), self.iter_num)
        for i in list(acc_dict.keys()):
            self.writer.add_scalar('val/class' + str(i), acc_dict[i], self.iter_num)
        #     self.acc_list[int(i)] += acc_dict[i]
        # if i_iter % self.epoch_iter == 0 and i_iter != 0:
        #     self.epoch_num += 1
        #     for i in list(acc_dict.keys()):
        #         self.writer.add_scalar('val/class_epoch' + str(i), self.acc_list[int(i)]/self.epoch_iter, self.epoch_num)
            
        #     self.acc_list = [0] * 26
        if self.config.dingding:
            self.message(self.config.cluster + self.config.train_info + self.config.time_str, self.iter_num,
                         str(pred_acc), str(pred_result),
                         str(acc_dict))

        clus_mapping = self.cluster_mapping  # mapping between source label and target cluster index

        uk_null = torch.ones_like(memo_pred_all).float().cuda() * uk_index
        map_mask = torch.zeros_like(memo_pred_all).float().cuda()

        for k, v in self.cluster_mapping.items():
            if v in self.global_label_set:
                map_mask += torch.where(memo_pred_all == k, torch.Tensor([1.0]).cuda().float(), map_mask.float())

        pred_label = torch.where(map_mask > 0, cls_pred_all, uk_null)

        gt_all = torch.where(gt_all >= self.config.num_classes, torch.Tensor([uk_index]).cuda(), gt_all.float())
        mask = pred_label != uk_index
        pred_binary = (pred_label == uk_index).squeeze().tolist()
        gt_binary = (gt_all == uk_index).squeeze().tolist()

        for i in gt_all.unique().tolist():
            mask = gt_all == i
            count = mask.sum().float()
            correct = (pred_label == gt_all) * mask
            correct = correct.sum().float()
            acc_dict[i] = ((correct / count).item(), count.item())
        accs.update(acc_dict)

        acc = np.mean(list(accs.avg.values()))
        self.print_acc(accs.avg)
        if uk_index not in accs.avg:
            self.model.train(True)
            # self.neptune_metric('memo-val/Test Accuracy[center]', acc)
            return acc, acc, 0.0, 0.0, 0.0
        bi_rec = metrics.recall_score(gt_binary, pred_binary, zero_division=0)
        bi_prec = metrics.precision_score(gt_binary, pred_binary, zero_division=0)
        # self.neptune_metric('val/bi recall[center]', bi_rec)
        # self.neptune_metric('val/bi prec[center]', bi_prec)

        k_acc = (acc * len(accs.avg) - accs.avg[uk_index]) / (len(accs.avg) - 1)
        uk_acc = accs.avg[uk_index]
        common_sum = 0.0
        common_cnt = 0.0
        for k, v in accs.sum.items():
            if k != uk_index:
                common_sum += v
                common_cnt += accs.count[k]
        common_acc = common_sum / common_cnt
        print('common_acc', common_acc)
        h_score = 2 * (common_acc * uk_acc) / (common_acc + uk_acc)
        # self.neptune_metric('memo-val/H-score', h_score)
        self.model.train(True)
        # self.neptune_metric('memo-val/Test Accuracy[center]', acc)
        # self.neptune_metric('memo-val/UK classification accuracy[center]', accs.avg[uk_index])
        # self.neptune_metric('memo-val/Known category accuracy[center]', k_acc)
        return acc, k_acc, h_score, bi_rec, bi_prec

    def get_src_centers(self):
        self.model.eval()
        num_cls = self.config.cls_share + self.config.cls_src

        if self.config.model != 'res50':
            s_center = torch.zeros((num_cls, self.config.embed_size)).float().cuda()
        else:
            s_center = torch.zeros((num_cls, self.config.embed_size)).float().cuda()
        if not self.config.bottleneck:
            s_center = torch.zeros((num_cls, self.config.embed_size)).float().cuda()

        counter = torch.zeros((num_cls, 1)).float().cuda()
        s_feats = []
        s_labels = []
        for _, batch in tqdm(enumerate(self.src_loader)):
            acc_dict = {}
            img, label, _ = batch
            label = label.cuda().squeeze()
            with torch.no_grad():
                _, neck, _, _, _, _, _ = self.model(img.cuda())
            neck = F.normalize(neck, p=2, dim=-1)
            N, C = neck.shape
            # print('label type', type(label))
            s_labels.extend(label.tolist())
            s_feats.extend(torch.chunk(neck, N, dim=0))
        s_feats = torch.stack(s_feats).squeeze()
        s_labels = torch.from_numpy(np.array(s_labels)).cuda()
        print(s_labels.shape, s_feats.shape, 's_labels, s_feats')
        for i in s_labels.unique():
            i_msk = s_labels == i
            index = i_msk.squeeze().nonzero(as_tuple=False)
            # print(s_feats.shape)
            i_feat = s_feats[index, :].mean(0)
            # print(i_feat.shape)
            i_feat = F.normalize(i_feat, p=2, dim=1)
            # print(i_feat.shape)
            i = i.type(torch.long)
            # print(i)
            s_center[i, :] = i_feat

        # print(s_center.shape, s_feats.shape, s_labels.shape, 's_center, s_feats, s_labels')
        return s_center, s_feats, s_labels

    def sklearn_kmeans(self, feat, num_centers, init=None):
        print('sklearn_kmeans start !!__')
        if self.config.cluster != 'KMeans':
            return self.faiss_kmeans(feat, num_centers, init=init)
        if init is not None:
            kmeans = KMeans(n_clusters=num_centers, init=init, random_state=0).fit(feat.cpu().numpy())
        else:
            kmeans = KMeans(n_clusters=num_centers, random_state=0).fit(feat.cpu().numpy())
        center, t_codes = kmeans.cluster_centers_, kmeans.labels_
        # center, t_codes = kmeans.affinity_matrix_, kmeans.labels_
        print('t_codes:', t_codes)
        score = sklearn.metrics.silhouette_score(feat.cpu().numpy(), t_codes)
        return torch.from_numpy(center).cuda(), torch.from_numpy(t_codes).cuda(), score

    def faiss_kmeans(self, feat, K, init=None, niter=500):
        print('faiss_kmeans__start')
        import faiss
        feat = feat.cpu().numpy()
        d = feat.shape[1]
        kmeans = faiss.Kmeans(d, K, niter=niter, verbose=False, spherical=True)
        kmeans.train(feat)
        center = kmeans.centroids
        D, I = kmeans.index.search(feat, 1)
        center = torch.from_numpy(center).cuda()
        I = torch.from_numpy(I).cuda()
        D = torch.from_numpy(D).cuda()
        center = F.normalize(center, p=2, dim=-1)
        print('center:', center.shape, 'I:', I.shape, 'D:', D.shape)
        return center, I.squeeze(), D
