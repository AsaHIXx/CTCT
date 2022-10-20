import glob
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

###################################
####### reduce dimension to X######
###################################
ensemble = '/data/home/scv4524/run/xx/ANN/dataset/gene_ensemble_18856.txt'
symbol = '/data/home/scv4524/run/xx/ANN/gene_hygo_18856.txt'


class Dimension_reduce(object):
    def __init__(self, source_arr, variance, mean, save_path, target_arr):
        self.source_arr = source_arr,
        self.var = variance,
        self.save_path = save_path
        self.target_arr = target_arr
        self.mean = mean

    def dimension_reduce(self, arr):
        """

        Args:
            arr: input data matices
            n: variance bound

        Returns: low dimension dataframe

        """
        gene_list = pd.read_csv(ensemble, sep='\t', names=['gene'])
        gene_index = list(gene_list['gene'])
        arr = np.asarray(arr)
        arr = arr.squeeze()
        assert len(arr.shape) == 2
        data_df = pd.DataFrame(arr)
        data_df.columns = gene_index
        vt = VarianceThreshold(threshold=self.var)
        df = data_df.T
        X_vt = pd.DataFrame(vt.fit_transform(data_df), columns=data_df.columns[vt.get_support()])
        feature_dimension = X_vt.shape[1]
        gene_selected = list(X_vt.columns)
        _mean_df = df.mean(axis=1)
        _mean_threshold = df[_mean_df >= self.mean]
        gene_cross_merge = []
        for gene in list(_mean_threshold.index):
            if gene in gene_selected:
                gene_cross_merge.append(gene)
        print('[feature_dimension, len(list(_mean_threshold.index)), len(gene_cross_merge)]',
              [feature_dimension, len(list(_mean_threshold.index)), len(gene_cross_merge)])
        return gene_cross_merge, [feature_dimension, len(list(_mean_threshold.index))], data_df.T

    @staticmethod
    def gene_list_save(gene, path):
        df = pd.DataFrame({'gene': gene})
        df.to_csv((path + '/gene_list.txt'), index=False, header=False)
        print('Gene list save to {}, dimension'.format(path))
        return None

    def feature_selection(self):
        feat_d_source, gene_selected_s, source_df = self.dimension_reduce(self.source_arr)
        feat_d_target, gene_selected_t, target_df = self.dimension_reduce(self.target_arr)
        gene_selected = [i for i in gene_selected_t if i not in gene_selected_s]
        gene_selected.extend(gene_selected_s)
        print('gene feature length', len(gene_selected))
        df_target = target_df[gene_selected]
        df_source = source_df[gene_selected]
        print('source shape {} and target shape {} ?!'.format(df_source.shape, df_target.shape))
        gene_df = pd.DataFrame({'gene': gene_selected})
        self.gene_list_save(gene_df, self.save_path)
        return df_target.values, df_source.values, len(gene_selected), gene_df


##################################################
####### Find Features with mean and variance######
##################################################

class VarianceMeanFeatureSelection(object):
    def __init__(self, path, variance_threshold=10, mean_threshold=5, file_short='/*Lognormal.csv', log=False):

        self.var = variance_threshold
        self.path = path
        self.mean = mean_threshold
        self.file = []
        self.log = log
        self.file_short = file_short
        self._file_read_in_list()

    def _file_read_in_list(self):
        self.file = []
        for file in sorted(glob.glob(self.path + self.file_short)):
            self.file.append(file)
        print('file list:', self.file)
        return self.file

    def dimension_reduce(self, df):
        """
        inner merge between variance and mean selection
        Args:
            arr: input data matices
            n: variance bound

        Returns: low dimension dataframe

        """
        if self.log:
            df = df.applymap(self.log_func)
        # vt = VarianceThreshold(threshold=self.var)
        # data_df = df.T
        # X_vt = pd.DataFrame(vt.fit_transform(data_df), columns=data_df.columns[vt.get_support()])
        # feature_dimension = X_vt.shape[1]
        # gene_selected = list(X_vt.columns)
        _var_df = df.var(axis=1)
        _var_threshold = df[_var_df >= self.var]
        gene_selected = list(_var_threshold.index)
        _mean_df = df.mean(axis=1)
        _mean_threshold = df[_mean_df >= self.mean]
        gene_cross_merge = []
        for gene in list(_mean_threshold.index):
            if gene in gene_selected:
                gene_cross_merge.append(gene)
        print('[feature_dimension, len(list(_mean_threshold.index)), len(gene_cross_merge)]',
              [len(list(_var_threshold.index)), len(list(_mean_threshold.index)), len(gene_cross_merge)])
        return gene_cross_merge, [len(list(_var_threshold.index)), len(list(_mean_threshold.index))]

    @staticmethod
    def log_func(x):
        return np.log2(x + 1)

    def feature_selection(self):
        """
        gene outer merge
        """
        gene_merge = []
        for file in self.file:
            print(file, 'in process')
            if file.split('.')[-1] in ['csv', 'xls']:
                df_temp = pd.read_csv(file, sep=',', index_col=0)
            else:
                df_temp = pd.read_csv(file, sep='\t', index_col=0)
            gene_cross_merge, _ = self.dimension_reduce(df_temp)
            for i in gene_cross_merge:
                if i not in gene_merge:
                    gene_merge.append(i)

        return gene_merge, len(gene_merge)

    @staticmethod
    def reshape(df, genes_):
        """

        Args:
            df:  dataframe
            gene_list: list of gene name/ensemble

        Returns: dataframe after reshape

        """
        # genes_list = pd.read_csv(gene_list, names=['gene'])
        # genes_ = genes_list['gene']
        index_ = pd.DataFrame({'gene': genes_})
        df.index.name = 'gene'
        exp_reshape = pd.merge(df, index_['gene'], how='right', left_on='gene', right_on='gene')
        exp_reshape = exp_reshape.fillna(0)
        exp_reshape.set_index('gene', inplace=True)
        print("after reshape, the matrix's dim is", exp_reshape.shape)
        return exp_reshape


class BaseGeneCountsFileProcesser(object):
    def __init__(self, path, file_short, gene_cross,
                 gene_path='/data/home/scv4524/run/xx/ANN/dataset/gene_ensemble_18856.txt'):
        self.path = path
        self.file_short = file_short
        self.gene_path = gene_path
        self.gene_cross = gene_cross
        self.file_list = []
        self._file_list_read_in()
        self._gene_df_read_in()

    def _file_list_read_in(self):
        for file in sorted(glob.glob(self.path + self.file_short)):
            self.file_list.append(file)
        print('file list:', self.file_list)
        return self.file_list

    def _gene_df_read_in(self):
        self.gene_df = pd.read_csv(self.gene_path, names=['gene'])
        self.gene_array = list(self.gene_df['gene'])
        return self.gene_df, self.gene_array

    @staticmethod
    def reshape(df, genes_):
        """

        Args:
            df:  dataframe
            gene_list: list of gene name/ensemble

        Returns: dataframe after reshape

        """
        index_ = pd.DataFrame({'gene': genes_})
        df.index.name = 'gene'
        exp_reshape = pd.merge(df, index_['gene'], how='right', left_on='gene', right_on='gene')
        exp_reshape = exp_reshape.fillna(0)
        exp_reshape.set_index('gene', inplace=True)
        print("after reshape, the matrix's dim is", exp_reshape.shape)
        return exp_reshape


class MultiCountsFileReshape(BaseGeneCountsFileProcesser):
    # def __init__(self, path, file_short,gene_cross,gene_path='/data/home/scv4524/run/xx/ANN/dataset/gene_ensemble_18856.txt'):
    #     super().__init__(path, file_short, gene_path)
    #     self.gene_cross = gene_cross

    def matrix_merge(self):
        gene_total = pd.DataFrame()
        for file in self.file_list:
            df_tmp = pd.read_csv(file, index_col=0)
            df_reshape = self.reshape(df_tmp, self.gene_cross)
            gene_total = pd.concat([gene_total, df_reshape], axis=1)
        return gene_total
