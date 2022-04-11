from __future__ import division
from __future__ import print_function
import sys
import pickle
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
sys.path.append('../../')
from gcn import GCNSynthetic
from utils.utils import normalize_adj
import datasets

# Hyper-parameters for synthetic models
hidden = 20
dropout = 0.0

# Note: the trained model performance may differ due to random init of the weights
def evaluate_model(dataset, dataset_id):

    # Re-assemble model
    model = GCNSynthetic(nfeat=dataset.n_features, nhid=hidden, nout=hidden,
                         nclass=dataset.n_classes, dropout=dropout, task=dataset.task,
                         num_nodes=dataset.max_num_nodes)
    model.load_state_dict(torch.load("../models/gcn_3layer_{}.pt".format(dataset_id)))
    model.eval()  # Model testing mode

    train_idx_list, test_idx_list = dataset.split_tr_ts_idx()

    y_pred_list = []
    y_label_list = []

    for idx in test_idx_list:

        if dataset.task == "node-class":
            sub_adj, sub_feat, sub_labels, orig_idx, new_idx, num_nodes = dataset[idx]

        elif dataset.task == "graph-class":
            sub_adj, sub_feat, sub_labels, num_nodes = dataset[idx]

        norm_adj = normalize_adj(sub_adj)

        output = model(sub_feat, norm_adj)

        if dataset.task == "node-class":
            y_pred = torch.argmax(output, dim=1)
            y_pred = y_pred[new_idx]
            sub_labels = sub_labels[new_idx]

        elif dataset.task == "graph-class":
            y_pred = torch.argmax(output, dim=0)

        y_pred_list.append(y_pred)
        y_label_list.append(sub_labels)

    print(dataset_id)
    print("Accuracy: ", accuracy_score(y_pred_list, y_label_list))
    print("Precision: ", precision_score(y_pred_list, y_label_list, average=None))
    print("Recall: ", recall_score(y_pred_list, y_label_list, average=None))
    print()


if __name__ == "__main__":
    dataset_list = ["syn1", "syn4", "syn5", "MUTAG"]
    dataset_dict = {}

    for dataset in dataset_list:

        dataset_dict[dataset] = datasets.avail_datasets_dict[dataset](dataset)

    for k, v in dataset_dict.items():
        evaluate_model(v, k)
