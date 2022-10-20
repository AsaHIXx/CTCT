import h5py
import numpy as np
import collections
import scipy.sparse as sp_sparse
import tables
import pandas as pd
import scipy.sparse


class H5toCSV:
    def __init__(self, file_path: str, gene_list: str):
        """
        Convert h5 file to CSV file, (get expresion matrix)
        :param file_path: file path for h5 file
        :param gene_list: gene list for expresion matrix
        """
        self.file_path = file_path
        self.gene_list = gene_list

    def _get_h5_info(self):
        with h5py.File(self.file_path, 'r+') as f:
            data = f["matrix"]
            info = list(data)
            return info

    @staticmethod
    def name_reshape(lis):
        lis = lis.tolist()
        for i in range(len(lis)):
            lis[i] = str(lis[i]).split("'")[1]
        return np.array(lis)

    def get_matrix_from_h5(self):
        with tables.open_file(self.file_path, 'r') as f:
            mat_group = f.get_node(f.root, 'matrix')
            barcodes = f.get_node(mat_group, 'barcodes').read()
            barcodes = self.name_reshape(barcodes)
            # print(barcodes)  ## 细胞名字
            data = getattr(mat_group, 'data').read()  # 表达矩阵
            indices = getattr(mat_group, 'indices').read()
            indptr = getattr(mat_group, 'indptr').read()
            shape = getattr(mat_group, 'shape').read()  # 上面三个是稀疏矩阵的参数
            matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)  # 读入
            mat_group_2 = f.get_node(mat_group, 'features')
            gene_names = f.get_node(mat_group_2, 'name').read()  # 基因
            gene_names = self.name_reshape(gene_names)

            return barcodes, matrix, gene_names

    def sparse_read_in(self):
        barcodes, filtered_feature_bc_matrix, gene_names = self.get_matrix_from_h5()
        msc_dataset = pd.DataFrame.sparse.from_spmatrix(filtered_feature_bc_matrix)
        msc_dataset.columns = barcodes
        msc_dataset.index = gene_names
        print('data components', self._get_h5_info())
        print(self.file_path, 'data_shape is', msc_dataset.shape)

        return msc_dataset

    def select_gene(self):
        msc_dataset = self.sparse_read_in()
        genes_list = pd.read_csv(self.gene_list, names=['gene'])
        genes_ = genes_list['gene']
        index_ = pd.DataFrame({'gene': genes_})
        msc_dataset.index.name = 'gene'
        msc_reshape = pd.merge(msc_dataset, index_['gene'], how='right', left_on='gene', right_on='gene')
        msc_reshape.fillna(0)
        print('after reshape', msc_reshape.shape)
        return msc_reshape
