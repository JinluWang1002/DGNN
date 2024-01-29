import random
import numpy as np
import torch
import os
from torch_sparse import spspmm
from torch import LongTensor
from numpy import ndarray
from torch_sparse import SparseTensor

root = os.path.split(__file__)[0]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_accuracy(output, labels):
    if len(labels) == 0:
        return 1.0
    predict = output.max(1)[1].type_as(labels)
    correct = predict.eq(labels).double()
    correct = int(correct.sum())
    return correct / len(labels)


def get_mask(y, train_ratio=0.6, test_ratio=0.2, device=None):
    if device is None:
        device = torch.device("cpu")
    train_indexes = list()
    test_indexes = list()
    val_indexes = list()
    npy = y.cpu().numpy()

    def get_sub_mask(sub_x_indexes):
        np.random.shuffle(sub_x_indexes)
        sub_train_count = int(len(sub_x_indexes) * train_ratio)
        sub_test_count = int(len(sub_x_indexes) * test_ratio)
        sub_train_indexes = sub_x_indexes[0:sub_train_count]
        sub_test_indexes = sub_x_indexes[sub_train_count:sub_train_count + sub_test_count]
        sub_val_indexes = sub_x_indexes[sub_train_count + sub_test_count:]
        return sub_train_indexes, sub_test_indexes, sub_val_indexes

    def flatten_np_list(np_list):
        total_size = sum([len(item) for item in np_list])
        result = ndarray(shape=total_size)
        last_i = 0
        for item in np_list:
            result[last_i:last_i + len(item)] = item
            last_i += len(item)
        return np.sort(result)

    for class_id in np.unique(npy):
        indexes = np.argwhere(npy == class_id).flatten().astype(int)
        m, n, q = get_sub_mask(indexes)
        train_indexes.append(m)
        test_indexes.append(n)
        val_indexes.append(q)
    train_indexes = LongTensor(flatten_np_list(train_indexes)).to(device)
    test_indexes = LongTensor(flatten_np_list(test_indexes)).to(device)
    val_indexes = LongTensor(flatten_np_list(val_indexes)).to(device)
    return train_indexes, test_indexes, val_indexes


def adj_norm(adj):
    def sp_eye(n):
        indices = torch.Tensor([list(range(n)), list(range(n))])
        values = torch.FloatTensor([1.0] * n)
        return torch.sparse_coo_tensor(indices=indices, values=values, size=[n, n])  # 构建自环矩阵（单位）

    device = adj.device
    n = adj.shape[0]
    adj = adj + sp_eye(n).to(device)
    adj = adj.coalesce()  # 将稠密表示变为稀疏表示
    adj_indices = adj.indices()  # 获得非0元素的索引
    adj_values = adj.values()  # 获得非0元素的值
    d_values = torch.spmm(adj, torch.FloatTensor([[1.0]] * n).to(device)).pow(-0.5).flatten()
    d_indices = torch.tensor([list(range(n)), list(range(n))]).to(device)
    out_indices, out_values = spspmm(
        indexA=d_indices, valueA=d_values,
        indexB=adj_indices, valueB=adj_values,
        m=n, k=n, n=n
    )
    out_indices, out_values = spspmm(
        indexA=out_indices, valueA=out_values,
        indexB=d_indices, valueB=d_values,
        m=n, k=n, n=n
    )  # D(-1/2) A D (-1/2)
    return torch.sparse_coo_tensor(indices=out_indices, values=out_values, size=[n, n]).to(device)


def load_dataset(dataset_name, norm=True, device=None):
    if device is None:
        device = torch.device("cpu")
    assert dataset_name in ["cora", "pubmed", "citeseer", "chameleon", "squirrel", "computers", "photo"]
    path = os.path.join(root, "dataset", "{}.pt".format(dataset_name))
    data_dict = torch.load(path, map_location=device)
    feat, label, n, nfeat, nclass, adj = (
        data_dict["feat"], data_dict["label"], data_dict["n"], data_dict["nfeat"], data_dict["nclass"], data_dict["adj"]
    )
    if norm:
        adj = adj_norm(adj)
    return feat, label, n, nfeat, nclass, adj

def to_normalized_sparsetensor(edge_index, N, mode='DAD'):
    row, col = edge_index  # adj>0 的索引赋值给 row 和 col
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.set_diag() # pyg库的方法，将邻接矩阵对角线元素设置为1
    deg = adj.sum(dim=1).to(torch.float) # 计算dim=1维度的加和，生成度矩阵
    deg_inv_sqrt = deg.pow(-0.5)  # D（-1/2）
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # 将deg_inv_sqrt中的无穷大值替换为0
    if mode == 'DA':
        return deg_inv_sqrt.view(-1,1) * deg_inv_sqrt.view(-1,1) * adj  # .view(-1,1) 第二个维度上是1,第一个维度随之计算
    if mode == 'DAD':
        return deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)  # 不理解为什么要把deg_inv_sqrt弄成列向量。