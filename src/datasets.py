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

    def __init__(self, dataset_id, train_bool=True, train_ratio=0.1):

        # TODO: Add error if folder not present
        path = "../data/MUTAG/"

        # Sparse adjacency matrix
        sparse_np_adj = np.loadtxt(path + "/MUTAG_A.txt", delimiter=", ", dtype=int, unpack=True)
        # Features: use node labels as features for graph classification
        np_features = np.loadtxt(path + "/MUTAG_node_labels.txt", dtype=int, unpack=True)
        self.features = torch.LongTensor(np_features)
        # Graph indicator: tells which node belongs to which graph
        np_array_g_ind = np.loadtxt(path + "/MUTAG_graph_indicator.txt", dtype=int, unpack=True)
        # Graph labels
        np_labels = np.loadtxt(path + "/MUTAG_graph_labels.txt", dtype=int, unpack=True)
        self.labels = torch.LongTensor(np_labels)

        # adj contains all the single disconnected graphs, need to split them apart
        self.adj = to_dense_adj(torch.LongTensor(sparse_np_adj)).squeeze()

        # Group nodes by graph indicator
        graphs_df = pd.DataFrame(np_array_g_ind, columns=["Indicator"])
        self.nodes_by_graph_dict = graphs_df.groupby("Indicator").indices

        self.adj_by_graph_dict = {}
        self.feat_by_graph_dict = {}

        for graph_ind, node_array in self.nodes_by_graph_dict.items():
            self.adj_by_graph_dict[graph_ind] = self.adj[node_array, :][:, node_array]
            self.feat_by_graph_dict[graph_ind] = self.features[node_array]

        # The node features are their label
        self.n_features = 1
        self.n_classes = len(np.unique(self.labels))
        self.task = "graph-class"

    def __len__(self):
        return len(self.adj_by_graph_dict)

    def __getitem__(self, idx):

        # The graph idx start from 1
        sub_adj = self.adj_by_graph_dict[idx+1]
        sub_feat = self.feat_by_graph_dict[idx+1]
        sub_labels = self.labels[idx+1]

        return sub_adj, sub_feat, sub_labels

    def split_tr_ts_idx(self, train_ratio=0.8):

        num_graphs = len(self)
        num_train = int(num_graphs * train_ratio)
        idx = range(num_graphs)

        np.random.shuffle(idx)
        train_idx = idx[:num_train]
        test_idx = idx[num_train:]

        return train_idx, test_idx
