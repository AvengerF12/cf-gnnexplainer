import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from torch_geometric.utils import dense_to_sparse
import matplotlib.pyplot as plt
import argparse
import pickle
import pandas as pd
import torch


header = ["node_idx", "new_idx", "cf_adj", "sub_adj", "y_pred_orig", "y_pred_new_actual",
          "label", "num_nodes", "loss_graph_dist"]

def visualize(df, idx_cf, figsize=(20,15)):

    sub_adj = torch.Tensor(df["sub_adj"][idx_cf])
    cf_adj = torch.Tensor(df["cf_adj"][idx_cf])

    deleted_edges = sub_adj.int() - cf_adj.int()
    deleted_edges[deleted_edges == -1] = 0

    added_edges = cf_adj.int() - sub_adj.int()
    added_edges[added_edges == -1] = 0

    print("Target node: {}, label: {}".format(df["new_idx"][idx_cf], df["label"][idx_cf]))
    print("Graph distance loss: {}".format(df["loss_graph_dist"][idx_cf]))
    print("Original prediction: {}, new prediciton: {}"
          .format(df["y_pred_orig"][idx_cf], df["y_pred_new_actual"][idx_cf]))
    num_nodes = df["num_nodes"][idx_cf]
    print("Num of nodes: {}".format(num_nodes))

    sparse_sub_adj = dense_to_sparse(sub_adj)[0].T.numpy()
    sparse_deleted_edges = dense_to_sparse(deleted_edges)[0].T.numpy()
    sparse_added_edges = dense_to_sparse(added_edges)[0].T.numpy()

    adj_graph = nx.Graph()
    adj_graph.add_nodes_from(range(num_nodes))
    adj_graph.add_edges_from(sparse_sub_adj)

    plt.figure(figsize=figsize)

    # Keep same pos
    pos = graphviz_layout(adj_graph, root=df["new_idx"][idx_cf])

    # sub_adj visualization with removed edges
    nx.draw_networkx_edges(adj_graph, pos, sparse_sub_adj, edge_color="black")
    nx.draw_networkx_edges(adj_graph, pos, sparse_deleted_edges, edge_color="red")
    nx.draw_networkx_edges(adj_graph, pos, sparse_added_edges, edge_color="green")

    nx.draw_networkx_nodes(adj_graph, pos, range(num_nodes))
    nx.draw_networkx_nodes(adj_graph, pos, [df["new_idx"][idx_cf]], node_color="yellow")
    nx.draw_networkx_labels(adj_graph, pos)


def visualize_by_path(df_path, idx_cf):

    # Load CF examples
    with open(df_path, "rb") as f:
        cf_examples = pickle.load(f)
        num_cf_examples = len(cf_examples)
        df_prep = []

        for example in cf_examples:
            # Ignore examples for which generating a CF wasn't possible
            if example[0] != []:
                df_prep.append(example[0])

        df = pd.DataFrame(df_prep, columns=header)

    visualize(df, idx_cf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, help="Path of the file containing the cf")
    parser.add_argument('--idx_cf', default=0, help="Id of the cf to visualize")
    args = parser.parse_args()

    visualize_by_path(args.path, args.idx_cf)
    plt.show()
