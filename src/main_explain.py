from __future__ import division
from __future__ import print_function
import sys
sys.path.append('..')
import argparse
import pickle
import numpy as np
import time
import torch
from gcn import GCNSynthetic
from cf_explanation.cf_explainer import CFExplainer
from utils.utils import normalize_adj, get_neighbourhood, safe_open
from torch_geometric.utils import dense_to_sparse


def main_explain(dataset, hid_units=20, n_layers=3, dropout_r=0, seed=42, lr=0.005,
                 optimizer="SGD", n_momentum=0, beta=0.5, num_epochs=500,
                 edge_del=False, edge_add=False, delta=False, bernoulli=False, verbose=False):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True)

    # Import dataset from GNN explainer paper
    with open("../data/gnn_explainer/{}.pickle".format(dataset[:4]), "rb") as f:
        data = pickle.load(f)

    adj = torch.Tensor(data["adj"]).squeeze()       # Does not include self loops
    features = torch.Tensor(data["feat"]).squeeze()
    labels = torch.tensor(data["labels"]).squeeze()
    idx_train = torch.tensor(data["train_idx"])
    idx_test = torch.tensor(data["test_idx"])
    # Needed for pytorch-geo functions, returns a sparse representation:
    # Edge indices: row/columns of cells containing non-zero entries
    # Edge attributes: tensor containing edge's features
    edge_index = dense_to_sparse(adj)

    # Change to binary task: 0 if not in house, 1 if in house
    if dataset == "syn1_binary":
        labels[labels == 2] = 1
        labels[labels == 3] = 1

    # According to reparam trick from GCN paper
    norm_adj = normalize_adj(adj)

    # Set up original model, get predictions
    model = GCNSynthetic(nfeat=features.shape[1], nhid=hid_units, nout=hid_units,
                         nclass=len(labels.unique()), dropout=dropout_r)

    model.load_state_dict(torch.load("../models/gcn_3layer_{}.pt".format(dataset)))
    model.eval()
    output = model(features, norm_adj)
    y_pred_orig = torch.argmax(output, dim=1)

    if verbose:
        print("y_true counts: {}".format(np.unique(labels.numpy(), return_counts=True)))
        # Confirm model is actually doing something
        print("y_pred_orig counts: {}".format(np.unique(y_pred_orig.numpy(), return_counts=True)))

    # Get CF examples in test set
    test_cf_examples = []
    start = time.time()
    #Note: these are the nodes for which a cf is generated
    idx_test_sublist = idx_test[:]
    num_cf_found = 0

    for i, v in enumerate(idx_test_sublist):

        sub_adj, sub_feat, sub_labels, node_dict = \
            get_neighbourhood(int(v), edge_index, n_layers + 1, features, labels)
        new_idx = node_dict[int(v)]

        # Sanity check
        sub_adj_diag = torch.diag(sub_adj)
        if sub_adj_diag[sub_adj_diag != 0].any():
            raise RuntimeError("Self-connections on graphs are not allowed")

        # Check that original model gives same prediction on full graph and subgraph
        with torch.no_grad():
            sub_adj_pred = model(sub_feat, normalize_adj(sub_adj))[new_idx]

        if verbose:
            print("Output original model, full adj: {}".format(output[v]))
            print("Output original model, sub adj: {}".format(sub_adj_pred))


        # Need to instantitate new cf_model for each instance because size of P
        # changes based on size of sub_adj
        explainer = CFExplainer(model=model,
                                sub_adj=sub_adj,
                                sub_feat=sub_feat,
                                n_hid=hid_units,
                                dropout=dropout_r,
                                sub_labels=sub_labels,
                                y_pred_orig=y_pred_orig[v],
                                num_classes=len(labels.unique()),
                                beta=beta,
                                edge_del=edge_del,
                                edge_add=edge_add,
                                delta=delta,
                                bernoulli=bernoulli,
                                verbose=verbose)
        # If edge_add=True: learn new adj matrix directly, else: only remove existing edges

        cf_example = explainer.explain(node_idx=v, cf_optimizer=optimizer, new_idx=new_idx,
                                       lr=lr, n_momentum=n_momentum,
                                       num_epochs=num_epochs)

        test_cf_examples.append(cf_example)

        # Check if cf example is not empty
        if cf_example[0] != []:
            num_cf_found += 1

        if verbose:
            time_frmt_str = "Time for {} epochs of one example ({}/{}): {:.4f}min"
            print(time_frmt_str.format(num_epochs, i+1, len(idx_test_sublist),
                                       (time.time() - start)/60))

    print("Total time elapsed: {:.4f} mins".format((time.time() - start)/60))
    # Includes also empty examples!
    print("Number of CF examples found: {}/{}".format(num_cf_found, len(idx_test_sublist)))

    # Build path and save CF examples in test set
    format_path = "../results/{}"

    if not delta:
        # In the orig formulation edge_add does both operations
        if edge_add:
            format_path += "_add_del_orig"
        elif edge_del:
            format_path += "_del_orig"

    else:

        if edge_add:
            format_path += "_add"
        if edge_del:
            format_path += "_del"

        format_path += "_delta"

    if bernoulli:
        format_path += "_bernoulli/"
    else:
        format_path += "/"


    format_path += "{}/cf_examples_lr{}_beta{}_mom{}_epochs{}"

    dest_path = format_path.format(dataset, optimizer, lr, beta,
                                   n_momentum, num_epochs)

    with safe_open(dest_path, "wb") as f:
        pickle.dump(test_cf_examples, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='syn1')

    # Based on original GCN models -- do not change
    parser.add_argument('--hidden', type=int, default=20, help='Number of hidden units.')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of convolutional layers.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (between 0 and 1)')

    # For explainer
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for explainer')
    parser.add_argument('--optimizer', type=str, default="SGD", help='SGD or Adadelta')
    parser.add_argument('--n_momentum', type=float, default=0.0, help='Nesterov momentum')
    parser.add_argument('--beta', type=float, default=0.5, help='Tradeoff for dist loss')
    parser.add_argument('--num_epochs', type=int, default=500, help='Num epochs for explainer')
    parser.add_argument('--edge_add', action='store_true', default=False,
                        help='Include edge additions?')
    parser.add_argument('--edge_del', action='store_true', default=False,
                        help='Include edge deletions?')
    parser.add_argument('--delta', action='store_true', default=False,
                        help='Use delta formulation of the problem?')
    parser.add_argument('--bernoulli', action='store_true', default=False,
                        help='Use bernoulli-based approach to generate P?')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Activate verbose output?')

    args = parser.parse_args()

    main_explain(args.dataset, args.hidden, args.n_layers, args.dropout, args.seed,
                 args.lr, args.optimizer, args.n_momentum, args.beta, args.num_epochs,
                 args.edge_del, args.edge_add, args.delta, args.bernoulli, args.verbose)
