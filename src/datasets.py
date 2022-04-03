from __future__ import division
from __future__ import print_function
import sys
import os
sys.path.append('..')
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.utils import get_neighbourhood
from torch_geometric.utils import dense_to_sparse, to_dense_adj

# Works for all syn* datasets
class SyntheticDataset(Dataset):

    def __init__(self, dataset_id, n_layers):

        self.n_layers = n_layers

        with open("../data/gnn_explainer/{}.pickle".format(dataset_id[:4]), "rb") as f:

            full_dataset = pickle.load(f)

        self.train_idx = full_dataset["train_idx"]
        self.test_idx = full_dataset["test_idx"]
        self.complete_idx = self.train_idx + self.test_idx

        self.adj = torch.Tensor(full_dataset["adj"]).squeeze()
        self.features = torch.Tensor(full_dataset["feat"]).squeeze()
        self.labels = torch.LongTensor(full_dataset["labels"]).squeeze()

        # Needed for pytorch-geo functions, returns a sparse representation:
        # Edge indices: row/columns of cells containing non-zero entries
        # Edge attributes: tensor containing edge's features
        self.sparse_adj = dense_to_sparse(self.adj)

        self.n_features = self.features.shape[1]
        self.n_classes = len(self.labels.unique())
        self.task = "node-class"

    def __len__(self):
        return len(self.dataset_idx)

    def __getitem__(self, orig_idx):

        sub_adj, sub_feat, sub_labels, node_dict = \
            get_neighbourhood(int(orig_idx), self.sparse_adj, self.n_layers + 1, self.features,
                              self.labels)
        new_idx = node_dict[int(orig_idx)]

        return sub_adj, sub_feat, sub_labels, orig_idx, new_idx

    def split_tr_ts_idx(self, train_ratio=None):
        return self.train_idx, self.test_idx


class MUTAGDataset(Dataset):

    def __init__(self, dataset_id):

        # TODO: Add error if folder not present
        path = "../data/MUTAG/"

        # Sparse adjacency matrix
        sparse_np_adj = np.loadtxt(path + "/MUTAG_A.txt", delimiter=", ", dtype=int, unpack=True)

        # Features: use node labels as features for graph classification
        np_features = np.loadtxt(path + "/MUTAG_node_labels.txt", dtype=int, unpack=True)
        tensor_features = torch.LongTensor(np_features)
        self.features = torch.nn.functional.one_hot(tensor_features)

        # Graph indicator: tells which node belongs to which graph
        np_array_g_ind = np.loadtxt(path + "/MUTAG_graph_indicator.txt", dtype=int, unpack=True)

        # Graph labels
        np_labels = np.loadtxt(path + "/MUTAG_graph_labels.txt", dtype=int, unpack=True)
        self.labels = torch.LongTensor(np_labels)
        # Re-label -1 label to 0 to avoid problems with loss
        self.labels[self.labels == -1] = 0

        # adj contains all the single disconnected graphs, need to split them apart
        self.adj = to_dense_adj(torch.LongTensor(sparse_np_adj)).squeeze()
        # Need to remove the top row and the left most col since the nodes in dataset start from 1
        self.adj = self.adj[1:,:][:, 1:]

        # Group nodes by graph indicator
        graphs_df = pd.DataFrame(np_array_g_ind, columns=["Indicator"])
        self.nodes_by_graph_dict = graphs_df.groupby("Indicator").indices

        self.adj_by_graph_arr = []
        self.feat_by_graph_arr = []

        # Keep track of biggest graph for padding purposes
        self.max_num_nodes = 0

        for graph_ind, node_array in self.nodes_by_graph_dict.items():
            num_nodes = len(node_array)

            if num_nodes > self.max_num_nodes:
                self.max_num_nodes = num_nodes

            self.adj_by_graph_arr.append(self.adj[node_array, :][:, node_array])
            self.feat_by_graph_arr.append(self.features[node_array])

        # The node features are their label
        self.n_features = self.features.shape[1]
        self.n_classes = len(np.unique(self.labels))
        self.task = "graph-class"

    def __len__(self):
        return len(self.adj_by_graph_arr)

    def __getitem__(self, idx):

        cur_adj = self.adj_by_graph_arr[idx]
        num_nodes = cur_adj.shape[0]
        adj_padded = torch.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = cur_adj

        cur_feat = self.feat_by_graph_arr[idx]
        feat_padded = torch.zeros((self.max_num_nodes, self.n_features))
        feat_padded[:num_nodes, :self.n_features] = cur_feat

        label = self.labels[idx]

        return adj_padded, feat_padded, label

    def split_tr_ts_idx(self, train_ratio=0.8):

        num_graphs = len(self)
        num_train = int(num_graphs * train_ratio)
        idx = list(range(num_graphs))

        np.random.shuffle(idx)
        train_idx = idx[:num_train]
        test_idx = idx[num_train:]

        return train_idx, test_idx
