import h5py
import numpy as np
import collections
import scipy.sparse as sp_sparse
import tables
import pandas as pd
import scipy.sparse
import os
import scipy.sparse as sp_sparse


class H5toCSV:
    def __init__(self, file_path: str, gene_list: str, save: bool = False, save_path: str = '../../dataset'):
        """
        Convert h5 file to CSV file, (get expresion matrix)
        :param file_path: file path for h5 file
        :param gene_list: gene list for expresion matrix
        """
        self.file_path = file_path
        self.gene_list = gene_list
        self.save_path = save_path
        self.save = save

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
            data = getattr(mat_group, 'data').read()  
            indices = getattr(mat_group, 'indices').read()
            indptr = getattr(mat_group, 'indptr').read()
            shape = getattr(mat_group, 'shape').read()  
            matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)  
            mat_group_2 = f.get_node(mat_group, 'features')
            gene_names = f.get_node(mat_group_2, 'name').read()  
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
        msc_reshape = msc_reshape.fillna(0)
        msc_reshape.set_index('gene', inplace=True)
        print(self.file_path, 'after reshape', msc_reshape.shape)
        if self.save:
            if self.save_path[-1] != '/':
                self.save_path = self.save_path + '/'
            prefix_path = self.save_path + self.file_path.split('/')[-1].split('.')[0] + '.csv'
            msc_reshape.to_csv(prefix_path, sep=',')
        return msc_reshape


class Data_Reshape:
    def __init__(self, dataframe=None, data_path: str = None,
                 gene_list: str = '/data/home/scv4524/run/xx/ANN/dataset/gene_hygo_selected.txt', save: bool = False,
                 save_path: str = '/data/home/scv4524/run/xx/ANN/dataset'):
        """
        reshape for general dataframe for gene expression matrix
        :param data_path:  str
        :param gene_list:  str
        """
        self.data_path = data_path
        self.gene_list = gene_list
        self.save = save
        self.save_path = save_path
        self.dataframe = dataframe
        if self.data_path:
            self.file_read_in()

    def file_read_in(self):
        if self.data_path.split('.')[-1] in ['xlsx', 'csv']:
            exp_mat = pd.read_csv(self.data_path, sep=',')
        else:
            exp_mat = pd.read_csv(self.data_path, sep='\t')
        print(self.data_path, 'data_shape is ', exp_mat.shape)
        self.exp_mat = exp_mat
        return exp_mat

    def reshape(self):
        if self.dataframe:
            self.exp_mat = self.dataframe
        genes_list = pd.read_csv(self.gene_list, names=['gene'])
        genes_ = genes_list['gene']
        index_ = pd.DataFrame({'gene': genes_})
        self.exp_mat.index.name = 'gene'
        exp_reshape = pd.merge(self.exp_mat, index_['gene'], how='right', left_on='gene', right_on='gene')
        exp_reshape = exp_reshape.fillna(0)
        exp_reshape.set_index('gene', inplace=True)
        print('after reshape, the matrix dim is', exp_reshape.shape)
        if self.save:
            if self.save_path[-1] != '/':
                self.save_path = self.save_path + '/'
            prefix_path = self.save_path + self.file_path.split('/')[-1].split('.')[0] + '.csv'
            exp_reshape.to_csv(prefix_path, sep=',')
        return exp_reshape


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path, 'make success!!')

    else:
        print('path already exists！！！')
    return None


def gene_feature_selection(arr, gene_lis):
    ensemble = '/data/home/scv4524/run/xx/ANN/dataset/Seurat_logNormal/TiSCH_ensemble_4561.txt'
    gene_list = pd.read_csv(ensemble, sep='\t', names=['gene'])
    gene_index = list(gene_list['gene'])
    data_df = pd.DataFrame(arr)
    data_df.columns = gene_index
    gene_selected = gene_lis
    gene_selected = list(gene_selected['gene'])
    data_df = data_df[gene_selected]
    return data_df.values


class MatrixMaskByClassType:
    def __init__(self, mat, type_mat, mask_list):
        # assert type(mask_list) == list
        import numpy as np
        self.mat = mat
        self.type_mat = type_mat
        self.mask_list = mask_list

    @staticmethod
    def matrix_mask_idx(mat, mask_class_type):
        assert type(mask_class_type) == int
        idx = np.where(mat != mask_class_type)
        return idx

    def mat_masked(self):
        for i in self.mask_list:
            idx = np.where(self.type_mat != int(i))
            self.type_mat = self.type_mat[idx[0]]
            self.mat = self.mat[idx[0], :]
            print('After {} masked the matrix shape is {}'.format(i, self.mat.shape))
        return self.mat, self.type_mat


def sampling_by_class(mat, mat_species, sample_rate, sample_class, seeds=1234):
    import random
    random.seed(seeds)
    print('Before sample mat mat_species shape', mat.shape, mat_species.shape)
    sample_idx = []
    for class_num in list(sorted(set(mat_species))):
        idx_t = np.where(mat_species == int(class_num))
        # print(len(idx_t[0]))

        if class_num in sample_class:
            idx_t = random.sample(list(idx_t[0]), int(len(idx_t[0]) * sample_rate))
            # idx_t = list(idx_t[0])[0:int(len(idx_t[0]) * sample_rate)]
        else:
            idx_t = idx_t[0]
        sample_idx.extend(list(idx_t))
        print('After {} type the sample number is {}'.format(class_num, len(sample_idx)))
    sampled_matrix = mat[sample_idx, :]
    sampled_mat_species = mat_species[sample_idx]
    print("sampled_matrix, sampled_mat_species's shape is", sampled_matrix.shape, sampled_mat_species.shape)
    random.seed(1234)
    return sampled_matrix, sampled_mat_species


def get_new_label(label, remove_list):
    idx_remove = []
    for i in range(len(remove_list)):
        idx1 = np.argwhere(label == sorted(remove_list, reverse=True)[i])
        idx_remove.extend(list(idx1))
    for i in range(len(remove_list)):
        idx = np.where(label > sorted(remove_list, reverse=True)[i])
        label[idx] = label[idx] - 1
    # print(idx_remove)
    label = np.delete(label, idx_remove)
    return label


def sample_random_split(data, label, proportion, seed):
    from sklearn.model_selection import train_test_split
    _, X, _, y = train_test_split(data, label, test_size=proportion, random_state=seed)
    return X, y


def tabular_read_in(path):
    if path.split('.')[-1] == 'npy':
        
        tabular = np.load(path)
        tabular = tabular.astype(np.float32)
    elif path.split('.')[-1] == 'csv':
        tabular = pd.read_csv(path, index_col=0)
        tabular = tabular.values.T
        
    elif path.split('.')[-1] == 'txt':
        tabular = np.loadtxt(path, dtype=np.float32)
    elif path.split('.')[-1] == 'npz':
        cm = sp_sparse.load_npz(path)
        tabular = np.array(cm.todense()).astype(np.float32)
        
    return tabular
    