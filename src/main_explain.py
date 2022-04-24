from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append('..')
import argparse
import pickle
import numpy as np
import time
import torch
from models import GCNSynthetic, GraphAttNet
from cf_explanation.cf_explainer import CFExplainer
from utils.utils import get_neighbourhood, safe_open
from torch_geometric.utils import dense_to_sparse
import datasets


def main_explain(dataset_id, hid_units=20, n_layers=3, dropout_r=0, seed=42, lr=0.005,
                 optimizer="SGD", n_momentum=0, beta=0.5, num_epochs=500, cem_mode=None,
                 edge_del=False, edge_add=False, delta=False, bernoulli=False, cuda=False,
                 rand_init=True, verbose=False):

    cuda = cuda and torch.cuda.is_available()

#    np.random.seed(seed)
#    torch.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True)

    device = None

    if cuda:
        device = "cuda"
#        torch.cuda.manual_seed(seed)

    if cem_mode is not None and (edge_del or edge_add or delta or bernoulli):
        raise RuntimeError("The CEM implementation doesn't support the arguments: "
                           + "edge_del, edge_add, delta or bernoulli")

    # Import dataset
    if dataset_id in datasets.avail_datasets_dict:
        dataset = datasets.avail_datasets_dict[dataset_id](dataset_id, device=device)
    else:
        raise RuntimeError("Unsupported dataset")

    if dataset.task not in ["graph-class", "node-class"]:
        raise RuntimeError("Task not supported")

    # Set up original model
    if dataset.task == "node-class":
        model = GCNSynthetic(nfeat=dataset.n_features, nhid=hid_units, nout=hid_units,
                             nclass=dataset.n_classes, dropout=dropout_r)
    elif dataset.task == "graph-class":
        model = GraphAttNet(nfeat=dataset.n_features, nhid=hid_units, nout=hid_units,
                            nclass=dataset.n_classes, dropout=dropout_r)

    # Freeze weights in original model
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Load saved model parameters
    model.load_state_dict(torch.load("../models/gcn_3layer_{}.pt".format(dataset_id)))
    model.eval()

    if cuda:
        model = model.cuda()

    # Get CF examples in test set
    test_cf_examples = []
    start = time.time()
    # Note: these are the nodes for which explanations are generated
    _, test_idx_list = dataset.split_tr_ts_idx()
    num_cf_found = 0

    for i in test_idx_list:

        if dataset.task == "node-class":
            sub_adj, sub_feat, sub_labels, orig_idx, new_idx, num_nodes = dataset[i]

        elif dataset.task == "graph-class":
            sub_adj, sub_feat, sub_labels, num_nodes = dataset[i]

        sub_adj = sub_adj.expand(1, -1, -1)
        output = model(sub_feat, sub_adj)

        if dataset.task == "node-class":
            y_pred_orig = torch.argmax(output, dim=2)
        elif dataset.task == "graph-class":
            y_pred_orig = torch.argmax(output, dim=1)

        # Sanity check
        sub_adj_diag = torch.diagonal(sub_adj, dim1=-2, dim2=-1)
        if sub_adj_diag[sub_adj_diag != 0].any():
            raise RuntimeError("Self-connections on graphs are not allowed")


        # Need to instantitate new cf_model for each instance because size of P
        # changes based on size of sub_adj
        # Note: sub_labels is just 1 label for graph class
        explainer = CFExplainer(model=model,
                                cf_optimizer=optimizer,
                                lr=lr,
                                n_momentum=n_momentum,
                                sub_adj=sub_adj,
                                num_nodes=num_nodes,
                                sub_feat=sub_feat,
                                n_hid=hid_units,
                                dropout=dropout_r,
                                sub_labels=sub_labels,
                                num_classes=dataset.n_classes,
                                beta=beta,
                                task=dataset.task,
                                cem_mode=cem_mode,
                                edge_del=edge_del,
                                edge_add=edge_add,
                                delta=delta,
                                bernoulli=bernoulli,
                                rand_init=rand_init,
                                device=device,
                                verbose=verbose)

        if cuda:
            explainer.cf_model.cuda()

        if dataset.task == "node-class":

            cf_example = explainer.explain(task=dataset.task, y_pred_orig=y_pred_orig,
                                           node_idx=orig_idx, new_idx=new_idx,
                                           num_epochs=num_epochs)
        elif dataset.task == "graph-class":
            cf_example = explainer.explain(task=dataset.task, num_epochs=num_epochs,
                                           y_pred_orig=y_pred_orig)

        test_cf_examples.append(cf_example)

        # Check if cf example is not empty
        if cf_example[0] != []:
            num_cf_found += 1

        if verbose:
            time_frmt_str = "Time for {} epochs of one example ({}/{}): {:.4f}min"
            print(time_frmt_str.format(num_epochs, i+1, len(test_idx_list),
                                       (time.time() - start)/60))

    print("Total time elapsed: {:.4f} mins".format((time.time() - start)/60))
    # Includes also empty examples!
    print("Number of CF examples found: {}/{}".format(num_cf_found, len(test_idx_list)))

    # Build path and save CF examples in test set
    format_path = "../results/{}"

    if cem_mode is None:
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
            format_path += "_bernoulli"

        format_path += "/"
    else:
        format_path += "_" + cem_mode + "/"

    format_path += "{}/cf_examples_lr{}_beta{}_mom{}_epochs{}"

    if rand_init:
        format_path += "_rand"

    dest_path = format_path.format(dataset_id, optimizer, lr, beta, n_momentum, num_epochs)

    counter = 0
    # If a random init already exists, don't overwrite and create a new file
    while(rand_init):
        if not os.path.exists(dest_path + str(counter)):
            dest_path += str(counter)
            break
        else:
            counter += 1

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
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for explainer')
    parser.add_argument('--optimizer', type=str, default="SGD", help='SGD or Adadelta')
    parser.add_argument('--n_momentum', type=float, default=0.0, help='Nesterov momentum')
    parser.add_argument('--beta', type=float, default=0.5, help='Tradeoff for dist loss')
    parser.add_argument('--num_epochs', type=int, default=500, help='Num epochs for explainer')
    parser.add_argument('--cem_mode', type=str, default=None, help='PP/PN contrastive explanation')
    parser.add_argument('--edge_add', action='store_true', default=False,
                        help='Include edge additions?')
    parser.add_argument('--edge_del', action='store_true', default=False,
                        help='Include edge deletions?')
    parser.add_argument('--delta', action='store_true', default=False,
                        help='Use delta formulation of the problem?')
    parser.add_argument('--bernoulli', action='store_true', default=False,
                        help='Use bernoulli-based approach to generate P?')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Activate CUDA support?')
    parser.add_argument('--no_rand_init', action='store_true', default=False,
                        help='Disable random initialisation of the P matrix')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Activate verbose output?')

    args = parser.parse_args()

    main_explain(args.dataset, args.hidden, args.n_layers, args.dropout, args.seed, args.lr,
                 args.optimizer, args.n_momentum, args.beta, args.num_epochs, args.cem_mode,
                 args.edge_del, args.edge_add, args.delta, args.bernoulli, args.cuda,
                 not args.no_rand_init, args.verbose)
