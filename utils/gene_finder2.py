import scanpy as sc
import numpy as np
import pandas as pd
import datetime 
import os
import argparse
from data_preprocessing import tabular_read_in
parser = argparse.ArgumentParser(description='Gene marker finder')
parser.add_argument('--result_dir', type=str, default='results/ques2_')
parser.add_argument('--gene_file', type=str, default='data_examples/gene_hygo_18856.txt')
parser.add_argument('--source_arr', type=str, default='')
parser.add_argument('--target_arr', type=str, default='')
parser.add_argument('--target_labels', type=str, default='')
parser.add_argument('--source_labels', type=str, default='')
args = parser.parse_args()
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')
start_time = datetime.datetime.now()
time_str = start_time.strftime('%Y_%m_%d_%H_%M')
result_dir = args.result_dir + time_str
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path, 'make success!!')

    else:
        print('path already exists！！！')
    return None
mkdir(result_dir)


source_arr = tabular_read_in(args.source_arr)
target_arr = tabular_read_in(args.target_arr)
source_labels = tabular_read_in(args.source_labels)
target_labels = tabular_read_in(args.target_labels)
gene_file = args.gene_file

def sampling_by_class_1(mat, mat_species,sample_class):
    print('Before sample mat mat_species shape', mat.shape, mat_species.shape)
    idx_t = np.where(mat_species == sample_class)

        
    idx_t = idx_t[0]
    sampled_matrix = mat[idx_t, :]
    sampled_mat_species = mat_species[idx_t]
    print('class is ', sample_class)
    print("sampled_matrix, sampled_mat_species's shape is", sampled_matrix.shape, sampled_mat_species.shape)
    return sampled_matrix, sampled_mat_species

def sampling_by_class(mat, mat_species,sample_class):
    print('Before sample mat mat_species shape', mat.shape, mat_species.shape)
    idx_t = np.where(mat_species == sample_class)
    idx_t_anothoner = np.where(mat_species != sample_class)

        
    idx_t = idx_t[0]
    idx_n = idx_t_anothoner[0]
    arr_select = mat[idx_t, :]
    arr_last = mat[idx_n, :]    
    label_select = mat_species[idx_n]
    label_last = mat_species[idx_n]
    print('class is ', sample_class)
    print('arr_selected, arr_last shape is', arr_select.shape, arr_last.shape)
    return arr_select, arr_last, label_select, label_last



def return_marker_result_by_log(adata, names, n):
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    ctc_result  = pd.DataFrame(
        {names + '_' + key: result[key][names]
        for key in ['names', 'pvals','logfoldchanges']})

    sorted_ctc=ctc_result.sort_values(by=names+'_logfoldchanges', ascending=False).head(n)
    return sorted_ctc
def return_marker_result(adata, names):
    sc.tl.rank_genes_groups(adata, 'class', method='wilcoxon')
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    ctc_result  = pd.DataFrame(
        {names + '_' + key: result[key][names]
        for key in ['names', 'pvals','logfoldchanges']})

    sorted_ctc=ctc_result.sort_values(by=names+'_logfoldchanges', ascending=False)
    return sorted_ctc
def labels_generate(types, source_arr, target_arr, l=['source_', 'ctc_']):
    s = pd.DataFrame()
    labels = [l[0]+str(types) for i in range(source_arr.shape[0])]
    labels.extend([l[1]+str(types) for i in range(target_arr.shape[0])])
    s['class'] = labels
    return s
def return_type_samples(types, target_arr, target_labels, source_arr, source_labels):
    target_arr_ , _ = sampling_by_class_1(target_arr, target_labels, types)
    source_arr_ , _ = sampling_by_class_1(source_arr, source_labels, types)
    return target_arr_, source_arr_
def return_marker_results(source, s, var, types):
    adata = sc.AnnData(source, obs=s,var=var)

    source_ =  return_marker_result(adata, 'source_'+str(types))
    ctc = return_marker_result(adata, 'ctc_'+str(types))
    source_.to_csv(result_dir+'/ques_2_source_marker_class_{}_.csv'.format(types))
    ctc.to_csv(result_dir+'/ques_2_ctc_marker_class_{}_.csv'.format(types))



gene = pd.read_csv(gene_file, sep='\t', names=['gene'])
gene_symbol = list(gene['gene'])
var = pd.DataFrame(index= gene_symbol)
label_for_inves = list(set(target_labels))
for i in label_for_inves:
    target_,source_ = return_type_samples(i, target_arr, target_labels, source_arr, source_labels)
    
    s_ = labels_generate(i, source_, target_)
    
    total = np.append(source_, target_, axis=0)
    return_marker_results(total, s_, var, i)