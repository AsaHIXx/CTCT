import pandas as pd
import numpy as np
import glob
import argparse




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

def log_exp(path, gene_name):
    exp = pd.read_csv(path, index_col=0)
    exp = exp.applymap(log_func)
    exp_log = reshape(exp, gene_name)
    return exp_log
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate log tpm')
    parser.add_argument('--ensemble', type=str, default='./data_examples/gene_ensemble_18856.txt')
    parser.add_argument('--symbol', type=str, default='./data_examples/gene_hygo_18856.txt')
    parser.add_argument('--file', type=str, default='')
    parser.add_argument('--save_path', type=str, default='./data_examples/')
    args = parser.parse_args()
    filename = (args.file.split('/')[-1]).split('.')[0]
    ensemble = args.ensemble
    symbol = args.symbol
    exp_logtpm = log_exp(args.file)
    exp_logtpm.to_csv(args.savepath+filename+'logtpm.csv')