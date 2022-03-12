from __future__ import division
from __future__ import print_function
import sys
sys.path.append('../../')
import json
import argparse
import numpy as np
import os
import pandas as pd
import pickle
import torch
from torch_geometric.utils import dense_to_sparse
from gcn import GCNSynthetic
from utils.utils import normalize_adj, get_neighbourhood


res_path = "../results/"

header = ["node_idx", "new_idx", "cf_adj", "sub_adj", "y_pred_orig", "y_pred_new_actual",
          "label", "num_nodes", "loss_graph_dist"]
hidden = 20  # Hidden units per layer
dropout = 0.0  # Dropout disabled

for subdir, dirs, files in os.walk(res_path):
    for file in files:
        path = os.path.join(subdir, file)

        if ".ipynb" in path:
            continue

        if "syn4" in path:
            dataset = "syn4"
            dataset_name = "Tree-Cycles"
        elif "syn5" in path:
            dataset = "syn5"
            dataset_name = "Tree-Grid"
        elif "syn1" in path:
            dataset = "syn1"
            dataset_name = "BA-shapes"

        with open("../data/gnn_explainer/{}.pickle".format(dataset), "rb") as f:
            data = pickle.load(f)

        adj = torch.Tensor(data["adj"]).squeeze()  # Does not include self loops
        features = torch.Tensor(data["feat"]).squeeze()
        labels = torch.tensor(data["labels"]).squeeze()
        idx_train = torch.tensor(data["train_idx"])
        idx_test = torch.tensor(data["test_idx"])
        edge_index = dense_to_sparse(adj)

        norm_adj = normalize_adj(adj)
        model = GCNSynthetic(nfeat=features.shape[1], nhid=hidden, nout=hidden,
                             nclass=len(labels.unique()), dropout=dropout)
        model.load_state_dict(torch.load("../models/gcn_3layer_{}.pt".format(dataset)))
        model.eval() # Testing mode
        output = model(features, norm_adj)
        y_pred_orig = torch.argmax(output, dim=1)


        num_cf_examples = None
        # Load CF examples
        with open(path, "rb") as f:
            cf_examples = pickle.load(f)
            num_cf_examples = len(cf_examples)
            df_prep = []

            for example in cf_examples:
                # Ignore examples for which generating a CF wasn't possible
                if example[0] != []:
                    df_prep.append(example[0])
            df = pd.DataFrame(df_prep, columns=header)

        # Add num edges for each generated CF
        num_edges = []
        for i in df.index:

            num_edges.append(sum(sum(df["sub_adj"][i])) / 2)

        df["num_edges"] = num_edges

        # For accuracy, only look at motif nodes among CFs
        df_motif = df[df["y_pred_orig"] != 0].reset_index(drop=True)
        accuracy = []

        # Get original predictions
        dict_ypred_orig = dict(zip(sorted(np.concatenate((idx_train.numpy(), idx_test.numpy()))),
                                   y_pred_orig.numpy()))

        for i in range(len(df_motif)):
            node_idx = df_motif["node_idx"][i]
            new_idx = df_motif["new_idx"][i]
            _, _, _, node_dict = get_neighbourhood(int(node_idx), edge_index, 4, features, labels)

            # Confirm idx mapping is correct
            if node_dict[node_idx] == df_motif["new_idx"][i]:

                cf_adj = df_motif["cf_adj"][i]
                sub_adj = df_motif["sub_adj"][i]
                perturb = np.abs(cf_adj - sub_adj)
                perturb_edges = np.nonzero(perturb)  # Changed edge indices

                nodes_involved = np.unique(np.concatenate((perturb_edges[0], perturb_edges[1]),
                                                          axis=0))
                perturb_nodes = nodes_involved[nodes_involved != new_idx]  # Remove original node

                # Retrieve original node idxs for original predictions
                perturb_nodes_orig_idx = []
                for j in perturb_nodes:
                    list_comp = [key for (key, value) in node_dict.items() if value == j]
                    perturb_nodes_orig_idx.append(list_comp)

                perturb_nodes_orig_idx = np.array(perturb_nodes_orig_idx).flatten()

                # Retrieve original predictions
                perturb_nodes_orig_ypred = np.array([dict_ypred_orig[k] for k in perturb_nodes_orig_idx])
                nodes_in_motif = perturb_nodes_orig_ypred[perturb_nodes_orig_ypred != 0]
                prop_correct = len(nodes_in_motif) / len(perturb_nodes_orig_idx)

                accuracy.append([node_idx, new_idx, perturb_nodes_orig_idx,
                                 perturb_nodes_orig_ypred, nodes_in_motif, prop_correct])
            else:
                raise RuntimeError("Error in node mapping")

        df_accuracy = pd.DataFrame(accuracy, columns=["node_idx", "new_idx",
                                                      "perturb_nodes_orig_idx",
                                                      "perturb_nodes_orig_ypred",
                                                      "nodes_in_motif", "prop_correct"])



        print(path)
        print("Dataset name: {}".format(dataset_name))
        print("Num cf examples found: {}/{}".format(len(df), num_cf_examples))
        print("Avg fidelity: {}".format(1 - len(df) / num_cf_examples))
        print("Average graph distance: {}".format(np.mean(df["loss_graph_dist"])))
        print("Average sparsity: {}".format(np.mean(1 - df["loss_graph_dist"] / df["num_edges"])))
        print("Accuracy", np.mean(df_accuracy["prop_correct"]))
        print(" ")
        print("***************************************************************")
        print(" ")
