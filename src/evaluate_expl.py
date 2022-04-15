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
from models import GCNSynthetic, GraphAttNet
from utils.utils import normalize_adj, get_neighbourhood
import datasets


default_path = "../results/"

header = ["node_idx", "new_idx", "cf_adj", "sub_adj", "y_pred_orig", "y_pred_new_actual",
          "label", "num_nodes", "loss_graph_dist"]
hidden = 20
dropout = 0.0


def compute_edge_based_accuracy(df_motif, edge_index, features, labels, dict_ypred_orig, task):

    add_prop_correct = 0
    del_prop_correct = 0

    accuracy_edges = []

    # Compute accuracy
    for i in range(len(df_motif)):
        # These idxs are taken from the explainer output
        node_idx = df_motif["node_idx"][i]
        new_idx = df_motif["new_idx"][i]
        _, _, _, node_dict = get_neighbourhood(int(node_idx), edge_index, 4, features, labels)
        # Find orig_idx given new_idx
        node_dict_inv = {value: key for key, value in node_dict.items()}

        # Confirm idx mapping is correct
        if node_dict[node_idx] != df_motif["new_idx"][i]:
            raise RuntimeError("Error in node mapping")

        orig_adj = df_motif["sub_adj"][i]
        cf_adj = df_motif["cf_adj"][i]

        pert_del_edges = orig_adj - cf_adj
        pert_del_edges[pert_del_edges == -1] = 0

        pert_add_edges = cf_adj - orig_adj
        pert_add_edges[pert_add_edges == -1] = 0

        # Take into account only the lower half of the matrices since they are symmetric
        pert_del_edges = np.tril(pert_del_edges, k=-1)
        pert_add_edges = np.tril(pert_add_edges, k=-1)

        # Changed edge indices
        idx_del_edges = np.nonzero(pert_del_edges)
        idx_add_edges = np.nonzero(pert_add_edges)

        valid_del_edges = 0
        total_del_edges = 0
        for i in range(len(idx_del_edges[0])):
            node_1 = idx_del_edges[0][i]
            node_2 = idx_del_edges[1][i]

            orig_idx_1 = node_dict_inv[node_1]
            orig_idx_2 = node_dict_inv[node_2]

            pred_1 = dict_ypred_orig[orig_idx_1]
            pred_2 = dict_ypred_orig[orig_idx_2]

            if pred_1 != 0 and pred_2 != 0:
                valid_del_edges += 1

            total_del_edges += 1

        valid_add_edges = 0
        total_add_edges = 0
        for i in range(len(idx_add_edges[0])):
            node_1 = idx_add_edges[0][i]
            node_2 = idx_add_edges[1][i]

            orig_idx_1 = node_dict_inv[node_1]
            orig_idx_2 = node_dict_inv[node_2]

            pred_1 = dict_ypred_orig[orig_idx_1]
            pred_2 = dict_ypred_orig[orig_idx_2]

            # Note: here the accuracy uses "or" in order to allow connections between
            # the motif and a node outside it, allows for some flexibility by considering valid
            # edges used to build a new motif by addition
            if pred_1 != 0 or pred_2 != 0:
                valid_add_edges += 1

            total_add_edges += 1

        # Sanity check in case of no change to sub_adj
        # In case of PP it could be that the minimal explanation is the starting graph
        if total_del_edges == 0 and total_add_edges == 0 and task != "PP":
            raise RuntimeError("evaluate: sub_adj and cf_adj are identical")

        # Handle situation in which edges are only added and del accuracy is NaN
        if total_del_edges == 0:
            del_prop_correct = np.NaN

        else:
            del_prop_correct = valid_del_edges / total_del_edges

        if total_add_edges == 0:
            add_prop_correct = np.NaN

        else:
            add_prop_correct = valid_add_edges / total_add_edges

        accuracy_edges.append([node_idx, new_idx, del_prop_correct, add_prop_correct])

    return accuracy_edges


# Approach to compute accuracy used in paper, extended for edge addition
def compute_node_based_accuracy(df_motif, edge_index, features, labels, dict_ypred_orig, task):

    add_prop_correct = 0
    del_prop_correct = 0

    accuracy_nodes = []

    # Compute accuracy
    for i in range(len(df_motif)):
        # These idxs are taken from the explainer output
        node_idx = df_motif["node_idx"][i]
        new_idx = df_motif["new_idx"][i]
        _, _, _, node_dict = get_neighbourhood(int(node_idx), edge_index, 4, features, labels)
        # Find orig_idx given new_idx
        node_dict_inv = {value: key for key, value in node_dict.items()}

        # Confirm idx mapping is correct
        if node_dict[node_idx] != df_motif["new_idx"][i]:
            raise RuntimeError("Error in node mapping")

        orig_adj = df_motif["sub_adj"][i]
        cf_adj = df_motif["cf_adj"][i]

        pert_del_edges = orig_adj - cf_adj
        pert_del_edges[pert_del_edges == -1] = 0

        pert_add_edges = cf_adj - orig_adj
        pert_add_edges[pert_add_edges == -1] = 0

        # Changed edge indices
        idx_del_edges = np.nonzero(pert_del_edges)
        idx_add_edges = np.nonzero(pert_add_edges)

        # Find idx of nodes involved in the edge perturbations
        nodes_del_edges = np.unique(np.concatenate((idx_del_edges[0], idx_del_edges[1]), axis=0))
        nodes_add_edges = np.unique(np.concatenate((idx_add_edges[0], idx_add_edges[1]), axis=0))

        # Remove original node since we already know its prediction
        nodes_del_edges = nodes_del_edges[nodes_del_edges != new_idx]
        nodes_add_edges = nodes_add_edges[nodes_add_edges != new_idx]

        # Retrieve original node idxs for original predictions
        del_nodes_orig_idx = [node_dict_inv[new_idx] for new_idx in nodes_del_edges]
        add_nodes_orig_idx = [node_dict_inv[new_idx] for new_idx in nodes_add_edges]

        # Retrieve original predictions
        del_nodes_orig_ypred = np.array([dict_ypred_orig[k] for k in del_nodes_orig_idx])
        add_nodes_orig_ypred = np.array([dict_ypred_orig[k] for k in add_nodes_orig_idx])

        # Retrieve nodes in motif (ground truth)
        del_nodes_in_motif = del_nodes_orig_ypred[del_nodes_orig_ypred != 0]
        add_nodes_in_motif = add_nodes_orig_ypred[add_nodes_orig_ypred != 0]

        # Sanity check in case of no change to sub_adj
        # In case of PP it could be that the minimal explanation is the starting graph
        if len(del_nodes_orig_idx) == 0 and len(add_nodes_orig_idx) == 0 and task != "PP":
            raise RuntimeError("evaluate: sub_adj and cf_adj are identical")

        # Handle situation in which edges are only added and del accuracy is NaN
        if len(del_nodes_orig_idx) == 0:
            del_prop_correct = np.NaN

        else:
            del_prop_correct = len(del_nodes_in_motif) / len(del_nodes_orig_idx)

        if len(add_nodes_orig_idx) == 0:
            add_prop_correct = np.NaN

        else:
            add_prop_correct = len(add_nodes_in_motif) / len(add_nodes_orig_idx)

        accuracy_nodes.append([node_idx, new_idx, del_prop_correct, add_prop_correct])

    return accuracy_nodes


# The purpose of accuracy is to evaluate how minimal the explanations are wrt the motif
# learned by the underlying model
def compute_accuracy_measures(cf_df, dataset, dataset_id, task):

    # Load relevant data from dataset for model
    # Note: no self-connections in syn*
    adj = torch.Tensor(dataset["adj"]).squeeze()
    features = torch.Tensor(dataset["feat"]).squeeze()
    labels = torch.tensor(dataset["labels"]).squeeze()
    idx_train = torch.tensor(dataset["train_idx"])
    idx_test = torch.tensor(dataset["test_idx"])
    edge_index = dense_to_sparse(adj)

    # Re-assemble model
    model = GCNSynthetic(nfeat=features.shape[1], nhid=hidden, nout=hidden,
                         nclass=len(labels.unique()), dropout=dropout)
    model.load_state_dict(torch.load("../models/gcn_3layer_{}.pt".format(dataset_id)))
    model.eval()  # Model testing mode

    norm_adj = normalize_adj(adj)
    output = model(features, norm_adj)
    y_pred_orig = torch.argmax(output, dim=1)

    # For accuracy, only look at nodes that the model believes belong to the motif
    # Note: 0 is used to mean that the node isn't in the motif
    df_motif = cf_df[cf_df["y_pred_orig"] != 0].reset_index(drop=True)

    # Get original predictions
    dict_ypred_orig = dict(zip(sorted(np.concatenate((idx_train.numpy(), idx_test.numpy()))),
                               y_pred_orig.numpy()))

    accuracy_nodes = compute_node_based_accuracy(df_motif, edge_index, features,
                                                 labels, dict_ypred_orig, task)
    accuracy_edges = compute_edge_based_accuracy(df_motif, edge_index, features,
                                                 labels, dict_ypred_orig, task)

    df_accuracy_nodes = pd.DataFrame(accuracy_nodes, columns=["node_idx", "new_idx",
                                                              "del_prop_correct",
                                                              "add_prop_correct"])

    df_accuracy_edges = pd.DataFrame(accuracy_edges, columns=["node_idx", "new_idx",
                                                              "del_prop_correct",
                                                              "add_prop_correct"])
    return df_accuracy_nodes, df_accuracy_edges


def evaluate(expl_list, dataset_id, dataset_name, dataset_data, expl_task, accuracy_bool=True):

    # Explanation count
    num_tot_expl = None
    num_valid_expl = None

    # Build CF examples dataframe
    num_tot_expl = len(expl_list)
    df_prep = []

    for expl in expl_list:
        # Ignore elements for which generating an explanation wasn't possible
        if expl[0] == []:
            continue

        df_prep.append(expl[0])

    num_valid_expl = len(df_prep)
    expl_df = pd.DataFrame(df_prep, columns=header)

    # Add num edges for each generated explanation
    expl_df["num_edges_adj"] = expl_df["sub_adj"].transform(lambda x: np.sum(x)/2)
    expl_df["num_edges_expl"] = expl_df["cf_adj"].transform(lambda x: np.sum(x)/2)

    if accuracy_bool and "syn" in dataset_id:
        # Compute different accuracy metrics only for synthetic datasets
        accuracy_nodes_df, accuracy_edges_df = \
            compute_accuracy_measures(expl_df, dataset_data, dataset_id, expl_task)

    if expl_task == "PP":
        fidelity = num_valid_expl / num_tot_expl
    else:
        # PN and counterfactual case
        fidelity = 1 - num_valid_expl / num_tot_expl

    avg_graph_dist = np.mean(expl_df["loss_graph_dist"])
    avg_change_ratio = np.mean(expl_df["num_edges_expl"] / expl_df["num_edges_adj"])

    results = {"dataset": dataset_name,
               "num_valid_expl": num_valid_expl,
               "num_examples": num_tot_expl,
               "fidelity": fidelity,
               "avg_graph_dist": avg_graph_dist,
               "avg_change_ratio": avg_change_ratio}

    if accuracy_bool and "syn" in dataset_id:
        avg_del_accuracy_nodes = np.mean(accuracy_nodes_df["del_prop_correct"])
        avg_add_accuracy_nodes = np.mean(accuracy_nodes_df["add_prop_correct"])

        avg_del_accuracy_edges = np.mean(accuracy_edges_df["del_prop_correct"])
        avg_add_accuracy_edges = np.mean(accuracy_edges_df["add_prop_correct"])

        results["avg_del_accuracy_nodes"] = avg_del_accuracy_nodes
        results["avg_add_accuracy_nodes"] = avg_add_accuracy_nodes
        results["avg_del_accuracy_edges"] = avg_del_accuracy_edges
        results["avg_add_accuracy_edges"] = avg_add_accuracy_edges
    else:
        results["avg_del_accuracy_nodes"] = np.NaN
        results["avg_add_accuracy_nodes"] = np.NaN
        results["avg_del_accuracy_edges"] = np.NaN
        results["avg_add_accuracy_edges"] = np.NaN

    return results


def evaluate_path_content(res_path):

    result_list = []
    # MUTAG evaluation doesn't need the dataset since computing accuracy w/o ground truths
    # isn't useful
    dataset_list = ["syn1", "syn4", "syn5"]
    dataset_dict = {"MUTAG": None}

    for dataset in dataset_list:

        with open("../data/gnn_explainer/{}.pickle".format(dataset), "rb") as f:
            dataset_dict[dataset] = pickle.load(f)

    for subdir, dirs, files in os.walk(res_path):
        for file in files:
            path = os.path.join(subdir, file)

            # Skip irrelevant path
            if ".ipynb" in path or ".txt" in path or ".csv" in path:
                continue

            for cur_id in datasets.avail_datasets_dict:

                if cur_id in path:
                    dataset_id = cur_id
                    dataset_name = datasets.datasets_name_dict[dataset_id]
                    break

            if "PP" in path:
                expl_task = "PP"
            elif "PN" in path:
                expl_task = "PN"
            else:
                expl_task = "CF"

            with open(path, "rb") as f:
                generated_expl = pickle.load(f)

            res = evaluate(generated_expl, dataset_id, dataset_name, dataset_dict[dataset_id],
                           expl_task)
            res["path"] = path

            result_list.append(res)

    return result_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--res_path', default=default_path, help='Result path')
    args = parser.parse_args()

    result_list = evaluate_path_content(args.res_path)

    for res in result_list:
        for k, v in res.items():
            print(k, ": ", v)

        print("\n***************************************************************\n")
