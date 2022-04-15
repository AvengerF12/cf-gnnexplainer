import math
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from utils.utils import get_degree_matrix
from utils.utils import normalize_adj
from datasets import SyntheticDataset, MUTAGDataset
from models import GCNSynthetic, GraphAttNet
from sklearn.metrics import accuracy_score, precision_score, recall_score
import datasets

# Note: the model selection for the project has been already performed by the authors of
# GNNExplainer, this script is only used to perform the training and model evaluation

# Adapted from GNNExplainer paper in order to have similar results to CF-GNNExplainer
def train_graph_classifier(G_dataset, model, device, args):
    train_idx, test_idx = G_dataset.split_tr_ts_idx(train_ratio=args.train_ratio)

    train_labels = dataset.labels[train_idx]
    test_labels = dataset.labels[test_idx]

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    ypred = None

    for epoch in range(1, args.num_epochs + 1):
        begin_time = time.time()

        train_ypred = []
        avg_loss = 0

        # TODO: add mini-batch learning support
        for idx in train_idx:
            model.zero_grad()

            adj, feat, label, _ = dataset[idx]

            norm_adj = normalize_adj(adj, device=device)
            ypred = model(feat, norm_adj)
            loss = model.loss(ypred, label)
            avg_loss += loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            ypred_label = torch.argmax(ypred, axis=0)

            if device == "cuda":
                ypred_label = ypred_label.cpu()

            train_ypred.append(ypred_label)

        avg_loss /= len(train_idx)

        elapsed = time.time() - begin_time

        if epoch % 10 == 0:
            print(
                "epoch: ",
                epoch,
                "; loss: ",
                avg_loss,
                "; train_acc: ",
                accuracy_score(train_ypred, train_labels.cpu()),
                "; train_prec: ",
                precision_score(train_ypred, train_labels.cpu(), average=None),
                "{0:0.2f}".format(elapsed),
            )

    test_ypred = []

    for idx in test_idx:
        adj, feat, label, _ = dataset[idx]

        norm_adj = normalize_adj(adj, device=device)
        ypred = model(feat, norm_adj)

        ypred_label = torch.argmax(ypred, axis=0)

        if device == "cuda":
            ypred_label = ypred_label.cpu()

        test_ypred.append(ypred_label)

    print(
        "test_acc: ",
        accuracy_score(test_ypred, test_labels.cpu()),
        "; test_prec: ",
        precision_score(test_ypred, test_labels.cpu(), average=None),
    )

    torch.save(model.state_dict(), "../models/gcn_3layer_{}.pt".format(args.dataset))


def train_node_classifier(G_dataset, model, device, args):

    train_idx, test_idx = G_dataset.split_tr_ts_idx(train_ratio=args.train_ratio)

    # Train on whole adj since its a single connected component (valid for sun*, results may vary)
    # Note: here we are assuming that to train a node classification gcn we need the entire adj,
    # not just a neighbourhood
    adj = G_dataset.adj
    feat = G_dataset.features
    labels_train = G_dataset.labels[train_idx]
    labels_test = G_dataset.labels[test_idx]

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    ypred = None

    norm_adj = normalize_adj(adj, device=device)

    for epoch in range(1, args.num_epochs + 1):
        begin_time = time.time()
        model.zero_grad()

        if args.cuda:
            ypred = model(feat.cuda(), norm_adj.cuda())
        else:
            ypred = model(feat, norm_adj)

        ypred_train = ypred[train_idx, :]
        ypred_test = ypred[test_idx, :]

        if args.cuda:
            loss = model.loss(ypred_train, labels_train.cuda())
        else:
            loss = model.loss(ypred_train, labels_train)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        elapsed = time.time() - begin_time

        if epoch % 10 == 0:
            print(
                "epoch: ",
                epoch,
                "; loss: ",
                loss.item(),
                "; train_acc: ",
                accuracy_score(torch.argmax(ypred_train, axis=1), labels_train),
                "; test_acc: ",
                accuracy_score(torch.argmax(ypred_test, axis=1), labels_test),
                "; train_prec: ",
                precision_score(torch.argmax(ypred_train, axis=1), labels_train, average="macro"),
                "; test_prec: ",
                precision_score(torch.argmax(ypred_test, axis=1), labels_test, average="macro"),
                "; epoch time: ",
                "{0:0.2f}".format(elapsed),
            )

    torch.save(model.state_dict(), "../models/gcn_3layer_{}.pt".format(args.dataset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='syn1')

    # Based on original GCN models -- do not change
    parser.add_argument('--hidden', type=int, default=20, help='Number of hidden units.')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of convolutional layers.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (between 0 and 1)')
    parser.add_argument('--clip', type=float, default=2.0, help='Norm clipping value')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--train-ratio', type=float, default=0.9, help='Ratio of data used for tr')
    parser.add_argument('--cuda', action='store_true', default=False, help='Activate CUDA support?')

    args = parser.parse_args()

    if args.cuda:
        device = "cuda"
    else:
        device = None

    dataset = datasets.avail_datasets_dict[args.dataset](args.dataset, device=device)

    if dataset.task == "node-class":
        model = GCNSynthetic(dataset.n_features, args.hidden, args.hidden, dataset.n_classes,
                             args.dropout)
    elif dataset.task == "graph-class":
        model = GraphAttNet(dataset.n_features, args.hidden, args.hidden, dataset.n_classes,
                            args.dropout)

    if args.cuda:
        model = model.cuda()

    if dataset.task == "node-class":
        train_node_classifier(dataset, model, device, args)

    elif dataset.task == "graph-class":
        train_graph_classifier(dataset, model, device, args)
