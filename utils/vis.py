import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

matplotlib.rcParams['font.sans-serif'] = ['KaiTi']

def vis_transfor(test_cells, test_cell_after_trans, test_labels, predicted, label_list, border, border_ni,
                 method='TSNE',
                 tumor_name: str = None,
                 path='./', infor: str = 'a'):
    """
    visualization for transfer learning before/after UMAP feature mapping result
    :param test_cells: before transfer expression matrix
    :param test_cell_after_trans: after tranfer expression matrix
    :param test_labels: ground truth for sample label
    :param predicted: model predict result
    :param label_list: all cell label set
    :param border: source and target data border index
    :param tumor_name: cell label name
    :param path: save path
    :return: UMAP result
    """
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1, len(label_list)))
    color_z = {}
    # for i in range(len(label_list)):
    #     if i in list(set(test_labels[border:border_ni])):
    #         color_z.update({i: colors[i]})
    #     else:
    #         color_z.update({i: 'grey'})

    color_z = {i: colors[i] for i in range(len(label_list))}
    cell_list_chs = [str(i) for i in range(len(label_list))]
    target_ground_class_num = list(set(test_labels[border:border_ni]))
    print(target_ground_class_num)
    target_pred_class_num = list(set(predicted[border:border_ni]))
    print('target_pred_class_num', target_pred_class_num)
    new_batch_ground_class_num = list(set(test_labels[border_ni:]))
    new_batch_pred_class_num = list(set(test_labels[border_ni:]))
    if tumor_name != None:
        cell_list = pd.read_csv(tumor_name, sep='\s+', names=['index', 'CHS', 'HANJI'])
        cell_list_chs = list(cell_list['CHS'])

    from sklearn.manifold import TSNE
    def prepare_plot_idx(test_labels, predicted, cls_no):
        return [j for j, item in enumerate(predicted) if item == cls_no], \
               [j for j, item in enumerate(test_labels) if item == cls_no]

    # TSNE = umap.UMAP(n_neighbors=50, random_state=111, min_dist=0.5)
    if method == 'umap':
        TSNE = umap.UMAP(random_state=1234)
    elif method == 'TSNE':
        TSNE = TSNE(n_components=2, random_state=1234)
    # cells_embedded_source = TSNE.fit_transform(test_cells[0:border, :])
    # cells_embedded_target = TSNE.fit_transform(test_cells[border:, :])
    cells_embedded = TSNE.fit_transform(test_cells)
    cells_embedded_source = cells_embedded[0:border, :]
    cells_embedded_target = cells_embedded[border:border_ni, :]
    cells_embedded_NewBatch = cells_embedded[border_ni:, :]
    # cells_embedded_trans_source = TSNE.fit_transform(test_cell_after_trans[0:border, :])
    # cells_embedded_trans_target = TSNE.fit_transform(test_cell_after_trans[border:, :])
    cells_embedded_trans = TSNE.fit_transform(test_cell_after_trans)
    cells_embedded_trans_source = cells_embedded_trans[0:border, :]
    cells_embedded_trans_target = cells_embedded_trans[border:border_ni, :]
    cells_embedded_trans_NewBatch = cells_embedded_trans[border_ni:, :]

    idx_all_source, idx_g_all_source, idx_all_target, idx_g_all_target, idx_all_newbatch, idx_g_all_newindex = [], [], [], [], [], []
    for i in range(len(label_list)):
        idx, idx_g = prepare_plot_idx(test_labels[0:border], predicted[0:border], i)
        idx_all_source.append(idx)
        idx_g_all_source.append(idx_g)
        idx_t, idx_g_t = prepare_plot_idx(test_labels[border:border_ni], predicted[border:border_ni], i)
        idx_all_target.append(idx_t)
        idx_g_all_target.append(idx_g_t)
        idx_n, idx_g_n = prepare_plot_idx(test_labels[border_ni:], predicted[border_ni:], i)
        idx_all_newbatch.append(idx_n)
        idx_g_all_newindex.append(idx_g_n)
    fig = plt.figure(figsize=(30, 10))
    plt.subplot(131)
    for i in range(len(label_list)):
        plt.scatter(cells_embedded_source[idx_g_all_source[i], 0], cells_embedded_source[idx_g_all_source[i], 1], s=1,
                    color=color_z[i], marker='o',
                    label=cell_list_chs[i])
    for i in target_ground_class_num:
        i = int(i)
        plt.scatter(cells_embedded_target[idx_g_all_target[i], 0], cells_embedded_target[idx_g_all_target[i], 1], s=20,
                    marker='v', color=color_z[i],
                    label=cell_list_chs[i])
    for i in new_batch_ground_class_num:
        i = int(i)
        plt.scatter(cells_embedded_NewBatch[idx_g_all_newindex[i], 0],
                    cells_embedded_NewBatch[idx_g_all_newindex[i], 1], s=30,
                    color=color_z[i], marker='*',
                    label=cell_list_chs[i])
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.title("Before")
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.subplot(132)
    for i in range(len(label_list)):
        plt.scatter(cells_embedded_trans_source[idx_g_all_source[i], 0],
                    cells_embedded_trans_source[idx_g_all_source[i], 1], s=1,
                    color=color_z[i], marker='o',
                    label=cell_list_chs[i])
    for i in target_ground_class_num:
        i = int(i)
        plt.scatter(cells_embedded_trans_target[idx_g_all_target[i], 0],
                    cells_embedded_trans_target[idx_g_all_target[i], 1], s=20,
                    color=color_z[i], marker='v',
                    label=cell_list_chs[i])
    for i in new_batch_ground_class_num:
        i = int(i)
        plt.scatter(cells_embedded_trans_NewBatch[idx_g_all_newindex[i], 0],
                    cells_embedded_trans_NewBatch[idx_g_all_newindex[i], 1], s=30,
                    color=color_z[i], marker='*',
                    label=cell_list_chs[i])
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.title("After")
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.subplot(133)
    for i in range(len(label_list)):
        plt.scatter(cells_embedded_trans_source[idx_g_all_source[i], 0],
                    cells_embedded_trans_source[idx_g_all_source[i], 1], s=1,
                    color=color_z[i], marker='o',
                    label=cell_list_chs[i])
    for i in target_pred_class_num:
        i = int(i)
        plt.scatter(cells_embedded_trans_target[idx_all_target[i], 0],
                    cells_embedded_trans_target[idx_all_target[i], 1], s=10,
                    color=color_z[i], marker='v',
                    label=cell_list_chs[i])
    for i in new_batch_pred_class_num:
        i = int(i)
        plt.scatter(cells_embedded_trans_NewBatch[idx_all_newbatch[i], 0],
                    cells_embedded_trans_NewBatch[idx_all_newbatch[i], 1], s=15,
                    color=color_z[i], marker='*',
                    label=cell_list_chs[i])

    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.title("Pred")
    plt.xlabel('t-' + method + ' 1')
    plt.ylabel('t-' + method + ' 2')
    plt.savefig(path + infor + method + '_1.jpg', dpi=400)
    plt.show()
    # vis_plot(cells_embedded_source, cells_embedded_target, test_labels, border, idx_g_all_target, idx_g_all_source,
    #          cells_embedded_trans_source, cells_embedded_trans_target, label_list, path='./', infor='A')
    return cells_embedded, cells_embedded_trans


def vis_batch(test_cells, test_cell_after_trans, test_labels, border, border_ni, label_list: list = ['tumor', 'ctc'],
              path='./', infor: str = 'A'):
    """
    visualization for transfer learning before/after UMAP feature mapping result
    :param test_cells: before transfer expression matrix
    :param test_cell_after_trans: after tranfer expression matrix
    :param test_labels: ground truth for sample label

    :param label_list: all cell label set
    :param border: source and target data border index

    :param path: save path
    :return: UMAP result
    """
    print('_______start batch shift visualize______')
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1, len(label_list)))
    color_z = {i: colors[i] for i in range(len(label_list))}
    # color_z = ['#FFA07A', '#FFF68F', '#87CEFA']
    cell_list_chs = label_list
    target_ground_class_num = list(set(test_labels[border:border_ni]))

    print(target_ground_class_num)

    from sklearn.manifold import TSNE
    def prepare_plot_idx(test_labels, cls_no):
        return [j for j, item in enumerate(test_labels) if item == cls_no]

    # TSNE = umap.UMAP(n_neighbors=50, random_state=111, min_dist=0.5)
    # TSNE = umap.UMAP()
    TSNE = TSNE(n_components=2, random_state=1234)

    cells_embedded = TSNE.fit_transform(test_cells)
    cells_embedded_source = cells_embedded[0:border, :]
    cells_embedded_target = cells_embedded[border:border_ni, :]
    cells_embedded_target_NewBatch = cells_embedded[border_ni:, :]
    # cells_embedded_trans_source = TSNE.fit_transform(test_cell_after_trans[0:border, :])
    # cells_embedded_trans_target = TSNE.fit_transform(test_cell_after_trans[border:, :])
    cells_embedded_trans = TSNE.fit_transform(test_cell_after_trans)
    cells_embedded_trans_source = cells_embedded_trans[0:border, :]
    cells_embedded_trans_target = cells_embedded_trans[border:border_ni, :]
    cells_embedded_trans_NewBatch = cells_embedded_trans[border_ni:, :]

    def idx_get(test_labels):
        idx_g_all = []
        for i in range(len(label_list)):
            idx_g = prepare_plot_idx(test_labels, i)
            idx_g_all.append(idx_g)

        return idx_g_all
        # idx_g_t = prepare_plot_idx(test_labels[border:], i)
        # idx_g_all_target.append(idx_g_t)

    idx_g_all_source = idx_get(test_labels[0:border])
    idx_g_all_target = idx_get(test_labels[border:border_ni])
    idx_g_all_NewBatch = idx_get(test_labels[border_ni:])
    print('target_index', idx_g_all_target)
    fig = plt.figure(figsize=(20, 10))
    plt.subplot(121)

    plt.scatter(cells_embedded_source[idx_g_all_source[0], 0], cells_embedded_source[idx_g_all_source[0], 1],
                s=1,
                color=color_z[0], marker='o',
                label=cell_list_chs[0])
    # for i in target_ground_class_num:
    #     i = int(i)
    plt.scatter(cells_embedded_target[idx_g_all_target[1], 0], cells_embedded_target[idx_g_all_target[1], 1],
                s=10,
                marker='v', color=color_z[1],
                label=cell_list_chs[1])
    plt.scatter(cells_embedded_target_NewBatch[idx_g_all_NewBatch[2], 0],
                cells_embedded_target_NewBatch[idx_g_all_NewBatch[2], 1],
                s=1,
                marker='*', color=color_z[2],
                label=cell_list_chs[2])
    plt.legend(loc='best')
    plt.title("Before")
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.subplot(122)
    plt.scatter(cells_embedded_trans_source[idx_g_all_source[0], 0],
                cells_embedded_trans_source[idx_g_all_source[0], 1], s=1,
                color=color_z[0], marker='o',
                label=cell_list_chs[0])
    for i in target_ground_class_num:
        i = int(i)
        plt.scatter(cells_embedded_trans_target[idx_g_all_target[i], 0],
                    cells_embedded_trans_target[idx_g_all_target[i], 1], s=10,
                    color=color_z[i], marker='v',
                    label=cell_list_chs[i])
    plt.scatter(cells_embedded_trans_NewBatch[idx_g_all_NewBatch[2], 0],
                cells_embedded_trans_NewBatch[idx_g_all_NewBatch[2], 1],
                s=1,
                marker='*', color=color_z[2],
                label=cell_list_chs[2])
    plt.legend(loc='best')
    plt.title("After")
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(path + infor + 'TSNE_2.jpg', dpi=400)
    plt.show()


def vis_transfor_2(test_cells, test_cell_after_trans, test_labels, predicted, label_list, border, border_ni,
                   tumor_name: str = None,
                   path='./', infor: str = 'a'):
    """
    visualization for transfer learning before/after UMAP feature mapping result
    :param test_cells: before transfer expression matrix
    :param test_cell_after_trans: after tranfer expression matrix
    :param test_labels: ground truth for sample label
    :param predicted: model predict result
    :param label_list: all cell label set
    :param border: source and target data border index
    :param tumor_name: cell label name
    :param path: save path
    :return: UMAP result
    """
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1, len(label_list)))
    color_z = {}
    for i in range(len(label_list)):
        if i in list(set(test_labels[border:border_ni])):
            color_z.update({i: colors[i]})
        else:
            color_z.update({i: 'grey'})

    # color_z = {i: colors[i] for i in range(len(label_list))}
    cell_list_chs = [str(i) for i in range(len(label_list))]
    target_ground_class_num = list(set(test_labels[border:border_ni]))
    print(target_ground_class_num)
    target_pred_class_num = list(set(predicted[border:border_ni]))
    print('target_pred_class_num', target_pred_class_num)
    new_batch_ground_class_num = list(set(test_labels[border_ni:]))
    new_batch_pred_class_num = list(set(test_labels[border_ni:]))
    if tumor_name != None:
        cell_list = pd.read_csv(tumor_name, sep='\s+', names=['index', 'CHS', 'HANJI'])
        cell_list_chs = list(cell_list['CHS'])

    from sklearn.manifold import TSNE
    def prepare_plot_idx(test_labels, predicted, cls_no):
        return [j for j, item in enumerate(predicted) if item == cls_no], \
               [j for j, item in enumerate(test_labels) if item == cls_no]

    # TSNE = umap.UMAP(n_neighbors=50, random_state=111, min_dist=0.5)
    # TSNE = umap.UMAP(random_state=111)
    TSNE = TSNE(n_components=2, random_state=111)
    # cells_embedded_source = TSNE.fit_transform(test_cells[0:border, :])
    # cells_embedded_target = TSNE.fit_transform(test_cells[border:, :])
    cells_embedded = TSNE.fit_transform(test_cells)
    cells_embedded_source = cells_embedded[0:border, :]
    cells_embedded_target = cells_embedded[border:border_ni, :]
    cells_embedded_NewBatch = cells_embedded[border_ni:, :]
    # cells_embedded_trans_source = TSNE.fit_transform(test_cell_after_trans[0:border, :])
    # cells_embedded_trans_target = TSNE.fit_transform(test_cell_after_trans[border:, :])
    cells_embedded_trans = TSNE.fit_transform(test_cell_after_trans)
    cells_embedded_trans_source = cells_embedded_trans[0:border, :]
    cells_embedded_trans_target = cells_embedded_trans[border:border_ni, :]
    cells_embedded_trans_NewBatch = cells_embedded_trans[border_ni:, :]

    idx_all_source, idx_g_all_source, idx_all_target, idx_g_all_target, idx_all_newbatch, idx_g_all_newindex = [], [], [], [], [], []
    for i in range(len(label_list)):
        idx, idx_g = prepare_plot_idx(test_labels[0:border], predicted[0:border], i)
        idx_all_source.append(idx)
        idx_g_all_source.append(idx_g)
        idx_t, idx_g_t = prepare_plot_idx(test_labels[border:border_ni], predicted[border:border_ni], i)
        idx_all_target.append(idx_t)
        idx_g_all_target.append(idx_g_t)
        idx_n, idx_g_n = prepare_plot_idx(test_labels[border_ni:], predicted[border_ni:], i)
        idx_all_newbatch.append(idx_n)
        idx_g_all_newindex.append(idx_g_n)
    fig = plt.figure(figsize=(30, 10))
    plt.subplot(131)
    for i in range(len(label_list)):
        plt.scatter(cells_embedded_source[idx_g_all_source[i], 0], cells_embedded_source[idx_g_all_source[i], 1], s=1,
                    color=color_z[i], marker='o',
                    label=cell_list_chs[i])
    for i in target_ground_class_num:
        i = int(i)
        plt.scatter(cells_embedded_target[idx_g_all_target[i], 0], cells_embedded_target[idx_g_all_target[i], 1], s=20,
                    marker='v', color=color_z[i],
                    label=cell_list_chs[i])
    for i in new_batch_ground_class_num:
        i = int(i)
        plt.scatter(cells_embedded_NewBatch[idx_g_all_newindex[i], 0],
                    cells_embedded_NewBatch[idx_g_all_newindex[i], 1], s=30,
                    color=color_z[i], marker='*',
                    label=cell_list_chs[i])
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.title("Before")
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.subplot(132)
    for i in range(len(label_list)):
        plt.scatter(cells_embedded_trans_source[idx_g_all_source[i], 0],
                    cells_embedded_trans_source[idx_g_all_source[i], 1], s=1,
                    color=color_z[i], marker='o',
                    label=cell_list_chs[i])
    for i in target_ground_class_num:
        i = int(i)
        plt.scatter(cells_embedded_trans_target[idx_g_all_target[i], 0],
                    cells_embedded_trans_target[idx_g_all_target[i], 1], s=20,
                    color=color_z[i], marker='v',
                    label=cell_list_chs[i])
    for i in new_batch_ground_class_num:
        i = int(i)
        plt.scatter(cells_embedded_trans_NewBatch[idx_g_all_newindex[i], 0],
                    cells_embedded_trans_NewBatch[idx_g_all_newindex[i], 1], s=30,
                    color=color_z[i], marker='*',
                    label=cell_list_chs[i])
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.title("After")
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.subplot(133)
    for i in range(len(label_list)):
        plt.scatter(cells_embedded_trans_source[idx_g_all_source[i], 0],
                    cells_embedded_trans_source[idx_g_all_source[i], 1], s=1,
                    color=color_z[i], marker='o',
                    label=cell_list_chs[i])
    for i in target_pred_class_num:
        i = int(i)
        plt.scatter(cells_embedded_trans_target[idx_all_target[i], 0],
                    cells_embedded_trans_target[idx_all_target[i], 1], s=10,
                    color=color_z[i], marker='v',
                    label=cell_list_chs[i])
    for i in new_batch_pred_class_num:
        i = int(i)
        plt.scatter(cells_embedded_trans_NewBatch[idx_all_newbatch[i], 0],
                    cells_embedded_trans_NewBatch[idx_all_newbatch[i], 1], s=15,
                    color=color_z[i], marker='*',
                    label=cell_list_chs[i])

    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.title("Pred")
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(path + infor + 'TSNE_batch_transfer_mapping.jpg', dpi=400)
    plt.show()


def vis_transfor_3(test_cells, test_cell_after_trans, test_labels, predicted, label_list, border, border_ni,
                   method='TSNE',
                   tumor_name: str = None,
                   path='./', infor: str = 'a'):
    """
    visualization for transfer learning before/after UMAP feature mapping result
    :param test_cells: before transfer expression matrix
    :param test_cell_after_trans: after tranfer expression matrix
    :param test_labels: ground truth for sample label
    :param predicted: model predict result
    :param label_list: all cell label set
    :param border: source and target data border index
    :param tumor_name: cell label name
    :param path: save path
    :return: UMAP result
    """
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1, len(label_list)))
    color_z = {}
    # for i in range(len(label_list)):
    #     if i in list(set(test_labels[border:border_ni])):
    #         color_z.update({i: colors[i]})
    #     else:
    #         color_z.update({i: 'grey'})

    color_z = {i: colors[i] for i in range(len(label_list))}
    cell_list_chs = [str(i) for i in range(len(label_list))]
    target_ground_class_num = list(set(test_labels[border:border_ni]))
    print(target_ground_class_num)
    target_pred_class_num = list(set(predicted[border:border_ni]))
    print('target_pred_class_num', target_pred_class_num)
    if tumor_name != None:
        cell_list = pd.read_csv(tumor_name, sep='\s+', names=['index', 'CHS', 'HANJI'])
        cell_list_chs = list(cell_list['CHS'])

    from sklearn.manifold import TSNE
    def prepare_plot_idx(test_labels, predicted, cls_no):
        return [j for j, item in enumerate(predicted) if item == cls_no], \
               [j for j, item in enumerate(test_labels) if item == cls_no]

    # TSNE = umap.UMAP(n_neighbors=50, random_state=111, min_dist=0.5)
    if method == 'umap':
        TSNE = umap.UMAP(random_state=1234)
    elif method == 'TSNE':
        TSNE = TSNE(n_components=2, random_state=1234)
    # cells_embedded_source = TSNE.fit_transform(test_cells[0:border, :])
    # cells_embedded_target = TSNE.fit_transform(test_cells[border:, :])
    cells_embedded = TSNE.fit_transform(test_cells)
    cells_embedded_source = cells_embedded[0:border, :]
    cells_embedded_target = cells_embedded[border:border_ni, :]
    # cells_embedded_trans_source = TSNE.fit_transform(test_cell_after_trans[0:border, :])
    # cells_embedded_trans_target = TSNE.fit_transform(test_cell_after_trans[border:, :])
    cells_embedded_trans = TSNE.fit_transform(test_cell_after_trans)
    cells_embedded_trans_source = cells_embedded_trans[0:border, :]
    cells_embedded_trans_target = cells_embedded_trans[border:border_ni, :]

    idx_all_source, idx_g_all_source, idx_all_target, idx_g_all_target, idx_all_newbatch, idx_g_all_newindex = [], [], [], [], [], []
    for i in range(len(label_list)):
        idx, idx_g = prepare_plot_idx(test_labels[0:border], predicted[0:border], i)
        idx_all_source.append(idx)
        idx_g_all_source.append(idx_g)
        idx_t, idx_g_t = prepare_plot_idx(test_labels[border:border_ni], predicted[border:border_ni], i)
        idx_all_target.append(idx_t)
        idx_g_all_target.append(idx_g_t)
    fig = plt.figure(figsize=(30, 10))
    plt.subplot(131)
    for i in range(len(label_list)):
        plt.scatter(cells_embedded_source[idx_g_all_source[i], 0], cells_embedded_source[idx_g_all_source[i], 1], s=1,
                    color=color_z[i], marker='o',
                    label=cell_list_chs[i])
    for i in target_ground_class_num:
        i = int(i)
        plt.scatter(cells_embedded_target[idx_g_all_target[i], 0], cells_embedded_target[idx_g_all_target[i], 1], s=20,
                    marker='v', color=color_z[i],
                    label=cell_list_chs[i])
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.title("Before")
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.subplot(132)
    for i in range(len(label_list)):
        plt.scatter(cells_embedded_trans_source[idx_g_all_source[i], 0],
                    cells_embedded_trans_source[idx_g_all_source[i], 1], s=1,
                    color=color_z[i], marker='o',
                    label=cell_list_chs[i])
    for i in target_ground_class_num:
        i = int(i)
        plt.scatter(cells_embedded_trans_target[idx_g_all_target[i], 0],
                    cells_embedded_trans_target[idx_g_all_target[i], 1], s=20,
                    color=color_z[i], marker='v',
                    label=cell_list_chs[i])
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.title("After")
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.subplot(133)
    for i in range(len(label_list)):
        plt.scatter(cells_embedded_trans_source[idx_g_all_source[i], 0],
                    cells_embedded_trans_source[idx_g_all_source[i], 1], s=1,
                    color=color_z[i], marker='o',
                    label=cell_list_chs[i])
    for i in target_pred_class_num:
        i = int(i)
        plt.scatter(cells_embedded_trans_target[idx_all_target[i], 0],
                    cells_embedded_trans_target[idx_all_target[i], 1], s=10,
                    color=color_z[i], marker='v',
                    label=cell_list_chs[i])
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.title("Pred")
    plt.xlabel('t-' + method + ' 1')
    plt.ylabel('t-' + method + ' 2')
    plt.savefig(path + infor + method + 'only_target_1.jpg', dpi=400)
    plt.show()
    return cells_embedded, cells_embedded_trans

