import os
import errno
import torch
import numpy as np
import pandas as pd
from torch_geometric.utils import k_hop_subgraph, dense_to_sparse, to_dense_adj, subgraph


# Used to implement Bernoulli rv approach to P generation in Srinivas paper (TODO)
class BernoulliMLSample(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):

        ctx.save_for_backward(input)

        output = torch.empty(input.shape)
        # ML sampling
        output[input >= 0.5] = 1
        output[input < 0.5] = 0

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Pass-through estimator of bernoulli
        return grad_output

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
    # Output: vector containing the sum for each row
    return torch.diag(sum(adj))

def normalize_adj(adj):
    # Normalize adjacancy matrix according to reparam trick in GCN paper
    A_tilde = adj + torch.eye(adj.shape[0])
    D_tilde = get_degree_matrix(A_tilde).detach()  # Don't need gradient
    # Raise to power -1/2, set all infs to 0s
    D_tilde_exp = D_tilde ** (-1 / 2)
    D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

    # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I)^(-1/2)
    norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

    return norm_adj

def get_neighbourhood(node_idx, edge_index, n_hops, features, labels):
    edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index[0])     # Get all nodes involved
    edge_subset_relabel = subgraph(edge_subset[0], edge_index[0], relabel_nodes=True)       # Get relabelled subset of edges
    sub_adj = to_dense_adj(edge_subset_relabel[0]).squeeze()
    sub_feat = features[edge_subset[0], :]
    sub_labels = labels[edge_subset[0]]
    new_index = np.array(range(len(edge_subset[0])))
    node_dict = dict(zip(edge_subset[0].numpy(), new_index))        # Maps orig labels to new
    # print("Num nodes in subgraph: {}".format(len(edge_subset[0])))
    return sub_adj, sub_feat, sub_labels, node_dict


# Note: the diagonal is assumed equal to 0
# the input vector contains everything below the diag
def create_symm_matrix_from_vec(vector, n_rows):
    matrix = torch.zeros(n_rows, n_rows)
    idx = torch.tril_indices(n_rows, n_rows, -1)
    matrix[idx[0], idx[1]] = vector
    symm_matrix = torch.tril(matrix) + torch.tril(matrix, -1).t()
    return symm_matrix


# Note: usual assumption about there being no self-connections in adj
def create_vec_from_symm_matrix(matrix, P_vec_size):
    idx = torch.tril_indices(matrix.shape[0], matrix.shape[0], -1)
    vector = matrix[idx[0], idx[1]]
    return vector
