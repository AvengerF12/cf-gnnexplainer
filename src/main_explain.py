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
                 optimizer="SGD", n_momentum=0, alpha=1, beta=0.5, gamma=0, num_epochs=500,
                 cem_mode=None, edge_del=False, edge_add=False, delta=False, bernoulli=False,
                 cuda=False, rand_init=True, history=True, hist_len=10, div_hind=5, verbosity=0):

    cuda = cuda and torch.cuda.is_available()

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True)

    device = None

    if cuda:
        device = "cuda"
        torch.cuda.manual_seed(seed)

    if cem_mode is not None and (edge_del or edge_add):
        raise RuntimeError("CEM implementation doesn't support the arguments: edge_del, edge_add")

    # Import dataset
    if dataset_id in datasets.avail_datasets_dict:
        dataset = datasets.avail_datasets_dict[dataset_id](dataset_id, device=device)
    else:
        raise RuntimeError("Unsupported dataset")

    if dataset.task not in ["graph-class", "node-class"]:
        raise RuntimeError("Task not supported")

    # Set up original model
    # Note: it is assumed that the models work with batches of data
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

    # Get explanations for data in test set
    test_expls = []
    start = time.time()
    # Note: these are the nodes for which explanations are generated
    _, test_idx_list = dataset.split_tr_ts_idx()
    num_expl_found = 0

    for i, v  in enumerate(test_idx_list):

        if dataset.task == "node-class":
            sub_adj, sub_feat, sub_labels, orig_idx, new_idx, num_nodes = dataset[v]
            sub_label = sub_labels[new_idx]

        elif dataset.task == "graph-class":
            sub_adj, sub_feat, sub_label, num_nodes = dataset[v]

        with torch.no_grad():
            output = model(sub_feat, sub_adj.expand(1, -1, -1)).squeeze()

            if dataset.task == "node-class":
                y_pred_orig = torch.argmax(output, dim=1)
                y_pred_orig = y_pred_orig[new_idx]
            elif dataset.task == "graph-class":
                y_pred_orig = torch.argmax(output, dim=0)

        # Sanity check
        sub_adj_diag = torch.diag(sub_adj)
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
                                sub_label=sub_label,
                                num_classes=dataset.n_classes,
                                alpha=alpha,
                                beta=beta,
                                gamma=gamma,
                                task=dataset.task,
                                cem_mode=cem_mode,
                                edge_del=edge_del,
                                edge_add=edge_add,
                                delta=delta,
                                bernoulli=bernoulli,
                                rand_init=rand_init,
                                history=history,
                                hist_len=hist_len,
                                div_hind=div_hind,
                                device=device,
                                verbosity=verbosity)

        if cuda:
            explainer.cf_model.cuda()

        if dataset.task == "node-class":

            expl, num_expl_inst = explainer.explain(task=dataset.task, y_pred_orig=y_pred_orig,
                                                    node_idx=orig_idx, new_idx=new_idx,
                                                    num_epochs=num_epochs)
        elif dataset.task == "graph-class":
            expl, num_expl_inst = explainer.explain(task=dataset.task, num_epochs=num_epochs,
                                                    y_pred_orig=y_pred_orig)

        test_expls.append(expl)

        # Count number of valid explanations generated
        if num_expl_inst > 0:
            num_expl_found += 1

        if verbosity > 0:
            time_frmt_str = "Time for {} epochs of one example ({}/{}): {:.4f}min"
            print(time_frmt_str.format(num_epochs, i + 1, len(test_idx_list),
                                       (time.time() - start)/60))

    print("Total time elapsed: {:.4f} mins".format((time.time() - start)/60))
    # Includes also empty examples!
    print("Number of CF examples found: {}/{}".format(num_expl_found, len(test_idx_list)))

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

    format_path += "{}/cf_examples_lr{}_alpha{}_beta{}_gamma{}_mom{}_epochs{}"

    if rand_init:
        format_path += "_rand"

    dest_path = format_path.format(dataset_id, optimizer, lr, alpha, beta, gamma,
                                   n_momentum, num_epochs)

    counter = 0
    # If a random init already exists, don't overwrite and create a new file
    while(rand_init):
        if not os.path.exists(dest_path + str(counter)):
            dest_path += str(counter)
            break
        else:
            counter += 1

    with safe_open(dest_path, "wb") as f:
        pickle.dump(test_expls, f)


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
    parser.add_argument('--alpha', type=float, default=1, help='Tradeoff for prediction loss')
    parser.add_argument('--beta', type=float, default=0.5, help='Tradeoff for dist loss')
    parser.add_argument('--gamma', type=float, default=0, help='Tradeoff for diversity loss')
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
    parser.add_argument('--no_history', action='store_true', default=False,
                        help='Store all the explanations generated during training?')
    parser.add_argument('--hist_len', type=int, default=10,
                        help='How long is the history kept for each explanation? ' + \
                        'If the history is longer than specified the result is a list of ' + \
                        'evenly-spaced elements of the original history.')
    parser.add_argument('--div_hind', type=int, default=5,
                        help='How many previous explanations to include when using diversity loss')
    parser.add_argument('--verbosity', type=int, default=0,
                        help='Level of output verbosity (0, 1, 2)')

    args = parser.parse_args()

    main_explain(args.dataset, args.hidden, args.n_layers, args.dropout, args.seed, args.lr,
                 args.optimizer, args.n_momentum, args.alpha, args.beta, args.gamma,
                 args.num_epochs, args.cem_mode, args.edge_del, args.edge_add, args.delta,
                 args.bernoulli, args.cuda, not args.no_rand_init, not args.no_history, args.hist_len,
                 args.div_hind, args.verbosity)
