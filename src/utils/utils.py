import os
import errno
import torch
import numpy as np
import pandas as pd
from torch_geometric.utils import k_hop_subgraph, dense_to_sparse, to_dense_adj, subgraph


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def safe_open(path, w):
    ''' Open "path" for writing, creating any parent directories as needed.'''
    mkdir_p(os.path.dirname(path))
    return open(path, w)


def get_degree_matrix(adj):
    return torch.diag(sum(adj))


def get_neighbourhood(node_idx, edge_index, n_hops, features, labels):
    edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index[0])  # Get all nodes involved
    edge_subset_relabel = subgraph(edge_subset[0], edge_index[0], relabel_nodes=True)  # Get relabelled subset of edges
    sub_adj = to_dense_adj(edge_subset_relabel[0]).squeeze()
    sub_feat = features[edge_subset[0], :]
    sub_labels = labels[edge_subset[0]]
    new_index = np.array(range(len(edge_subset[0])))
    node_dict = dict(zip(edge_subset[0].numpy(), new_index))  # Maps orig labels to new
    # print("Num nodes in subgraph: {}".format(len(edge_subset[0])))
    return sub_adj, sub_feat, sub_labels, node_dict


def create_symm_matrix_from_vec(vector, n_rows):
    matrix = torch.zeros(n_rows, n_rows)
    idx = torch.tril_indices(n_rows, n_rows)
    matrix[idx[0], idx[1]] = vector
    symm_matrix = torch.tril(matrix) + torch.tril(matrix, -1).t()
    return symm_matrix


def create_vec_from_symm_matrix(matrix, P_vec_size):
    idx = torch.tril_indices(matrix.shape[0], matrix.shape[0])
    vector = matrix[idx[0], idx[1]]
    return vector
