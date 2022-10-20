### QUES1
import scanpy as sc
import numpy as np
import pandas as pd
import datetime 
import os
import argparse
from data_preprocessing import tabular_read_in

parser = argparse.ArgumentParser(description='Gene marker finder')
parser.add_argument('--result_dir', type=str, default='results/TCGA_ques1_')
parser.add_argument('--gene_file', type=str, default='data_examples/gene_hygo_18856.txt')
parser.add_argument('--source_df_path', type=str, default='')
parser.add_argument('--label_path', type=str, default='')
args = parser.parse_args()

sc.settings.verbosity = 3       
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')
start_time = datetime.datetime.now()
time_str = start_time.strftime('%Y_%m_%d_%H_%M')
result_dir = args.result_dir + time_str
gene_file = args.gene_file
source_df_path = args.source_df_path
label_path = args.label_path
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path, 'make success!!')

    else:
        print('path already exists！！！')
    return None
mkdir(result_dir)

def log_func(x):
    return np.log2(x+1)

def reshape(df, gene_list):
    """

    Args:
        df:  dataframe
        gene_list: list of gene name/ensemble

    Returns: dataframe after reshape

    """
    genes_list = pd.read_csv(gene_list, names=['gene'])
    genes_ = genes_list['gene']
    index_ = pd.DataFrame({'gene': genes_})
    df.index.name = 'gene'
    print("before reshape, the matrix's dim is", df.shape)
    exp_reshape = pd.merge(df, index_['gene'], how='right', left_on='gene', right_on='gene')
    exp_reshape = exp_reshape.fillna(0)
    exp_reshape.set_index('gene', inplace=True)
    print("after reshape, the matrix's dim is", exp_reshape.shape)
    return exp_reshape


source_df = pd.read_csv(source_df_path, index_col=0)
source_df = source_df.applymap(log_func)
source_df  = reshape(source_df, gene_file)
source = source_df.values.T
label_df = pd.read_csv(label_path, index_col=0)
labels = list(label_df['condition'])
s = pd.DataFrame()
s['class'] = labels
label_classes = list(set(labels))




gene = pd.read_csv(gene_file, sep='\t', names=['gene'])
gene_symbol = list(gene['gene'])
# gene_symbol = list(source_df.index)
var = pd.DataFrame(index= gene_symbol)
adata = sc.AnnData(source, obs=s,var=var)
sc.tl.rank_genes_groups(adata, 'class', method='wilcoxon')

def return_marker_result_by_log(adata, names):
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    ctc_result  = pd.DataFrame(
        {names + '_' + key: result[key][names]
        for key in ['names', 'pvals','logfoldchanges']})

    sorted_ctc=ctc_result.sort_values(by=names+'_logfoldchanges', ascending=False)
    return sorted_ctc

for i in label_classes:
    source_sorted_marker = return_marker_result_by_log(adata=adata, names=str(i))
    source_sorted_marker.to_csv('./'+result_dir+'/ques_1_{}}_marker.csv'.format(str(i)))
