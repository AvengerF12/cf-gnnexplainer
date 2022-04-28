import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from torch_geometric.utils import dense_to_sparse
import matplotlib.pyplot as plt
import argparse
import pickle
import pandas as pd
import torch


header_data = ["node_idx", "new_idx", "expl_list", "sub_adj", "sub_feat", "label", "y_pred_orig",
               "num_nodes"]
# Structure of single element in expl_list
header_expl = ["cf_adj", "y_pred_new_actual", "loss_graph_dist"]

def visualize_mutag(df, idx_ex, idx_hist, figsize=(20,15)):

    mol_dict = {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"}
    node_labels = torch.argmax(df["sub_feat"][idx_ex], dim=1).numpy()
    cur_expl = df["expl_list"][idx_ex][idx_hist]

    sub_adj = torch.Tensor(df["sub_adj"][idx_ex])
    cf_adj = torch.Tensor(cur_expl[0])

    deleted_edges = sub_adj.int() - cf_adj.int()
    deleted_edges[deleted_edges == -1] = 0

    added_edges = cf_adj.int() - sub_adj.int()
    added_edges[added_edges == -1] = 0

    print("Graph label: {}".format(df["label"][idx_ex]))
    print("Graph distance loss: {}".format(cur_expl[2]))
    print("Original prediction: {}, new prediction: {}"
          .format(df["y_pred_orig"][idx_ex], cur_expl[1]))
    num_nodes = df["num_nodes"][idx_ex]
    print("Num of nodes: {}".format(num_nodes))
    num_edges = len(torch.nonzero(sub_adj)) / 2
    print("Num of total edges: {}".format(num_edges))

    sparse_sub_adj = dense_to_sparse(sub_adj)[0].T.numpy()
    sparse_deleted_edges = dense_to_sparse(deleted_edges)[0].T.numpy()
    sparse_added_edges = dense_to_sparse(added_edges)[0].T.numpy()

    adj_graph = nx.Graph()
    adj_graph.add_nodes_from(range(num_nodes))
    adj_graph.add_edges_from(sparse_sub_adj)

    plt.figure(figsize=figsize)

    # Keep same pos for consistent plotting
    pos = graphviz_layout(adj_graph)

    # sub_adj visualization with removed edges
    nx.draw_networkx_edges(adj_graph, pos, sparse_sub_adj, edge_color="black")
    nx.draw_networkx_edges(adj_graph, pos, sparse_deleted_edges, edge_color="red")
    nx.draw_networkx_edges(adj_graph, pos, sparse_added_edges, edge_color="green")

    nx.draw_networkx_nodes(adj_graph, pos, range(num_nodes))

    nx.draw_networkx_labels(adj_graph, pos, {k: mol_dict[node_labels[k]] for k in range(num_nodes)})


def visualize_generic(df, idx_ex, idx_hist, figsize=(20,15)):

    cur_expl = df["expl_list"][idx_ex][idx_hist]

    sub_adj = torch.Tensor(df["sub_adj"][idx_ex])
    cf_adj = torch.Tensor(cur_expl[0])

    deleted_edges = sub_adj.int() - cf_adj.int()
    deleted_edges[deleted_edges == -1] = 0

    added_edges = cf_adj.int() - sub_adj.int()
    added_edges[added_edges == -1] = 0

    print("Target node: {}, label: {}".format(df["new_idx"][idx_ex], df["label"][idx_ex]))
    print("Graph distance loss: {}".format(cur_expl[2]))
    print("Original prediction: {}, new prediction: {}"
          .format(df["y_pred_orig"][idx_ex], cur_expl[1]))
    num_nodes = df["num_nodes"][idx_ex]
    print("Num of nodes: {}".format(num_nodes))
    num_edges = len(torch.nonzero(sub_adj)) / 2
    print("Num of total edges: {}".format(num_edges))

    sparse_sub_adj = dense_to_sparse(sub_adj)[0].T.numpy()
    sparse_deleted_edges = dense_to_sparse(deleted_edges)[0].T.numpy()
    sparse_added_edges = dense_to_sparse(added_edges)[0].T.numpy()

    adj_graph = nx.Graph()
    adj_graph.add_nodes_from(range(num_nodes))
    adj_graph.add_edges_from(sparse_sub_adj)

    plt.figure(figsize=figsize)

    # Keep same pos for consistent plotting
    # In case of graph classification the idx of single nodes are not provided
    if df["new_idx"][idx_ex] is not None:
        pos = graphviz_layout(adj_graph, root=df["new_idx"][idx_ex])
    else:
        pos = graphviz_layout(adj_graph)

    # sub_adj visualization with removed edges
    nx.draw_networkx_edges(adj_graph, pos, sparse_sub_adj, edge_color="black")
    nx.draw_networkx_edges(adj_graph, pos, sparse_deleted_edges, edge_color="red")
    nx.draw_networkx_edges(adj_graph, pos, sparse_added_edges, edge_color="green")

    nx.draw_networkx_nodes(adj_graph, pos, range(num_nodes))

    if df["new_idx"][idx_ex] is not None:
        nx.draw_networkx_nodes(adj_graph, pos, [df["new_idx"][idx_ex]], node_color="yellow")

    nx.draw_networkx_labels(adj_graph, pos)


def visualize_by_path(df_path, idx_ex, idx_hist, dataset):

    # Load CF examples
    with open(df_path, "rb") as f:
        expls = pickle.load(f)
        df_prep = []

        for example in expls:
            # Ignore examples for which generating an explanation wasn't possible
            if example[2] != []:
                df_prep.append(example)

        df = pd.DataFrame(df_prep, columns=header_data)

    if dataset == "MUTAG":
        visualize_mutag(df, idx_ex, idx_hist)
    else:
        visualize_generic(df, idx_ex, idx_hist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None, help="Path of the file containing the cf")
    parser.add_argument('--idx_ex', type=int, default=0, help="Id of the explanation to visualize")
    parser.add_argument('--idx_hist', type=int, default=-1, help="Id of the element in history")
    parser.add_argument('--dataset', type=str, default=None,
                        help="Name of dataset of explanation. Specify for adhoc visualization.")
    args = parser.parse_args()

    visualize_by_path(args.path, args.idx_ex, args.idx_hist, args.dataset)
    plt.show()
