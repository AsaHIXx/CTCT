import numpy as np
import torch


def return_class_acc(pred, ground):
    """
    the accuracy of classes in valid dataset
    :param pred: pred result
    :param ground: ground truth
    :return: accuracy dict for every class
    """
    class_type = list(set(ground))
    acc_dict = {}
    arr1 = np.array(pred)
    arr2 = np.array(ground)
    for i in class_type:
        index_tmp = np.argwhere(arr2 == i)
        arr1_class = arr1[index_tmp]
        arr2_class = arr2[index_tmp]
        acc_class = np.mean(arr1_class == arr2_class)
        acc_dict.update({i: round(acc_class, 2)})
    return acc_dict


def one_hot_encoder(idx, n_cls):
    """
    generate one_hot encoding
    Parameters
    ----------
    idx
    n_cls

    Returns
    -------

    """
    assert torch.max(idx).item() < n_cls
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n_cls)
    onehot = onehot.to(idx.device)
    onehot.scatter_(1, idx.long(), 1)
    return onehot


def partition(data, partitions, num_partitions):
    """
    Hope to be completed
    :param data:
    :param partitions:
    :param num_partitions:
    :return:
    """
    res = []
    partitions = partitions.flatten()
    for i in range(num_partitions):
        indices = torch.nonzero((partitions == i), as_tuple=False).squeeze(1)
        res += [data[indices]]
    return res


def normalize_zscore(arr):
    """
    sklearn StandardScaler
    :param arr: numpy array input
    :return: normalize array
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data_scaler = scaler.fit_transform(arr)
    return data_scaler


def pca_reduce(arr, componets):
    """
    pca reduces dimension
    Args:
        arr: data
        componets: n_componets

    Returns: low dimension pca result

    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=componets)
    low_embedding = pca.fit_transform(arr)
    print('explained_variance:', pca.explained_variance_)
    return low_embedding


def classification_metric(pred_labels, true_labels):
    pred_labels = torch.ByteTensor(pred_labels)
    true_labels = torch.ByteTensor(true_labels)

    assert 1 >= pred_labels.all() >= 0
    assert 1 >= true_labels.all() >= 0

    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = torch.sum((pred_labels == 1) & ((true_labels == 1)))

    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = torch.sum((pred_labels == 0) & (true_labels == 0))

    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = torch.sum((pred_labels == 1) & (true_labels == 0))

    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = torch.sum((pred_labels == 0) & (true_labels == 1))
    return TP, TN, FP, FN


def find_duplciates1(lis):
    from collections import Counter
    result = dict(Counter(lis))
    idx_l = []

    key_list = [key for key, value in result.items() if value > 0]

    for i in key_list:
        idx = [j for j, x in enumerate(lis) if x == i]
        idx_l.append(idx)

    re_dict = dict(zip(key_list, idx_l))
    duplicates_dict = {key: value for key, value in result.items() if value > 1}

    return duplicates_dict, re_dict