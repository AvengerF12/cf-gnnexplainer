from __future__ import division
from __future__ import print_function
import sys
import pickle
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
sys.path.append('../../')
from torch.utils.data import SubsetRandomSampler, DataLoader
from models import GCNSynthetic, GraphAttNet
import datasets

# Hyper-parameters for synthetic models
hidden = 20
dropout = 0.0

# Note: the trained model performance may differ due to random init of the weights
def evaluate_model(dataset, dataset_id):

    # Re-assemble model
    if dataset.task == "node-class":
        model = GCNSynthetic(nfeat=dataset.n_features, nhid=hidden, nout=hidden,
                             nclass=dataset.n_classes, dropout=dropout)
    elif dataset.task == "graph-class":
        model = GraphAttNet(nfeat=dataset.n_features, nhid=hidden, nout=hidden,
                            nclass=dataset.n_classes, dropout=dropout)

    model.load_state_dict(torch.load("../models/gcn_3layer_{}.pt".format(dataset_id)))
    model.eval()  # Model testing mode

    train_idx_list, test_idx_list = dataset.split_tr_ts_idx()

    ts_idx_sampler = SubsetRandomSampler(test_idx_list)
    ts_dataloader = DataLoader(dataset, sampler=ts_idx_sampler)

    y_pred_list = []
    y_label_list = []

    for idx, data in enumerate(ts_dataloader):

        if dataset.task == "node-class":
            sub_adj, sub_feat, sub_labels, orig_idx, new_idx, num_nodes = data

        elif dataset.task == "graph-class":
            sub_adj, sub_feat, sub_labels, num_nodes = data

        output = model(sub_feat, sub_adj)

        if dataset.task == "node-class":
            y_pred = torch.argmax(output, dim=2)
            y_pred = y_pred[:, new_idx]
            sub_labels = sub_labels[:, new_idx]

        elif dataset.task == "graph-class":
            y_pred = torch.argmax(output, dim=1)

        y_pred_list.append(y_pred)
        y_label_list.append(sub_labels)

    print(dataset_id)
    print("Accuracy: ", accuracy_score(y_pred_list, y_label_list))
    print("Precision: ", precision_score(y_pred_list, y_label_list, average=None))
    print("Recall: ", recall_score(y_pred_list, y_label_list, average=None))
    print()


if __name__ == "__main__":
    dataset_list = ["MUTAG", "syn1", "syn4", "syn5"]
    dataset_dict = {}

    for dataset in dataset_list:

        dataset_dict[dataset] = datasets.avail_datasets_dict[dataset](dataset)

    for k, v in dataset_dict.items():
        evaluate_model(v, k)
