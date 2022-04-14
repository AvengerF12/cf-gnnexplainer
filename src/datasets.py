from __future__ import division
from __future__ import print_function
import sys
import os
sys.path.append('..')
import pickle
import numpy as np
from numpy.random import SeedSequence, default_rng
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.utils import get_neighbourhood
from torch_geometric.utils import dense_to_sparse, to_dense_adj


# Works for all syn* datasets
class SyntheticDataset(Dataset):

    def __init__(self, dataset_id, device=None, n_layers=3):

        self.dataset_id = dataset_id
        self.n_layers = n_layers

        with open("../data/gnn_explainer/{}.pickle".format(dataset_id[:4]), "rb") as f:

            full_dataset = pickle.load(f)

        self.train_idx = full_dataset["train_idx"]
        self.test_idx = full_dataset["test_idx"]
        self.complete_idx = self.train_idx + self.test_idx

        self.adj = torch.Tensor(full_dataset["adj"]).squeeze()
        self.features = torch.Tensor(full_dataset["feat"]).squeeze()
        self.labels = torch.LongTensor(full_dataset["labels"]).squeeze()

        if device == "cuda":
            self.adj = self.adj.cuda()
            self.features = self.features.cuda()
            self.labels = self.labels.cuda()

        # Needed for pytorch-geo functions, returns a sparse representation:
        # Edge indices: row/columns of cells containing non-zero entries
        # Edge attributes: tensor containing edge's features
        self.sparse_adj = dense_to_sparse(self.adj)

        self.n_features = self.features.shape[1]
        self.n_classes = len(self.labels.unique())
        self.task = "node-class"
        # The k-hop graphs have all the same number of nodes, this var is only for compatibility
        # It is only used in graph-class tasks
        self.max_num_nodes = None

    def __len__(self):
        return len(self.complete_idx)

    def __getitem__(self, orig_idx):

        sub_adj, sub_feat, sub_labels, node_dict = \
            get_neighbourhood(int(orig_idx), self.sparse_adj, self.n_layers + 1, self.features,
                              self.labels)
        new_idx = node_dict[int(orig_idx)]
        num_nodes = sub_adj.shape[1]

        return sub_adj, sub_feat, sub_labels, orig_idx, new_idx, num_nodes

    # Fixed split
    def split_tr_ts_idx(self, train_ratio=None):
        return self.train_idx, self.test_idx


class MUTAGDataset(Dataset):

    def __init__(self, dataset_id, device=None):

        self.dataset_id = dataset_id
        path = "../data/MUTAG/"

        # Sparse adjacency matrix
        sparse_np_adj = np.loadtxt(path + "/MUTAG_A.txt", delimiter=", ", dtype=int, unpack=True)

        # Features: use node labels as features for graph classification
        np_features = np.loadtxt(path + "/MUTAG_node_labels.txt", dtype=int, unpack=True)
        tensor_features = torch.LongTensor(np_features)
        self.features = torch.nn.functional.one_hot(tensor_features)
        self.features = self.features.type(torch.FloatTensor)

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

        if device == "cuda":
            # Load data
            self.adj = self.adj.cuda()
            self.features = self.features.cuda()
            self.labels = self.labels.cuda()

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
        self.n_classes = len(torch.unique(self.labels))
        self.task = "graph-class"

    def __len__(self):
        return len(self.adj_by_graph_arr)

    def __getitem__(self, idx):

        cur_adj = self.adj_by_graph_arr[idx]
        num_nodes = cur_adj.shape[0]
        nodes_diff = abs(self.max_num_nodes - num_nodes)

        # Pad bottom and right
        pad_adj_f = torch.nn.ZeroPad2d((0, nodes_diff, 0, nodes_diff))
        adj_padded = pad_adj_f(cur_adj)

        # Pad bottom
        pad_feat_f = torch.nn.ZeroPad2d((0, 0, 0, nodes_diff))
        cur_feat = self.feat_by_graph_arr[idx]
        feat_padded = pad_feat_f(cur_feat)

        # Scalar, no need to pad
        label = self.labels[idx]
        num_nodes = cur_adj.shape[1]

        return adj_padded, feat_padded, label, num_nodes

    def split_tr_ts_idx(self, train_ratio=0.9, seed=42):
        # Default split is:
        # - 0: 60, 1: 109 for tr set
        # - 0: 3, 1: 16 for ts set

        num_graphs = len(self)
        num_train = int(num_graphs * train_ratio)
        idx_list = list(range(num_graphs))
        rng_gen = default_rng(SeedSequence(seed).spawn(1)[0])

        rng_gen.shuffle(idx_list)
        train_idx = idx_list[:num_train]
        test_idx = idx_list[num_train:]

        return train_idx, test_idx


avail_datasets_dict = {"syn1": SyntheticDataset,
                       "syn4": SyntheticDataset,
                       "syn5": SyntheticDataset,
                       "MUTAG": MUTAGDataset}
datasets_name_dict = {"syn1": "BA-shapes (syn1)",
                      "syn4": "Tree-Cycles (syn4)",
                      "syn5": "Tree-Grid (syn5)",
                      "MUTAG": "Mutagenicity (MUTAG)"}
