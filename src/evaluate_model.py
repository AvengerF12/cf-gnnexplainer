from __future__ import division
from __future__ import print_function
import sys
import pickle
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
sys.path.append('../../')
from gcn import GCNSynthetic
from utils.utils import normalize_adj

# Hyper-parameters for synthetic models
hidden = 20
dropout = 0.0

# Note: the trained model performance may differ due to random init of the weights
def evaluate_model(dataset, dataset_id):

    # Note: no self-connections in syn*
    adj = torch.Tensor(dataset["adj"]).squeeze()
    features = torch.Tensor(dataset["feat"]).squeeze()
    labels = torch.tensor(dataset["labels"]).squeeze()

    # Re-assemble model
    norm_adj = normalize_adj(adj)
    model = GCNSynthetic(nfeat=features.shape[1], nhid=hidden, nout=hidden,
                         nclass=len(labels.unique()), dropout=dropout)
    model.load_state_dict(torch.load("../models/gcn_3layer_{}.pt".format(dataset_id)))
    model.eval()  # Model testing mode
    output = model(features, norm_adj)
    y_pred = torch.argmax(output, dim=1)

    print(dataset_id)
    print("Accuracy: ", accuracy_score(y_pred, labels))
    print("Precision: ", precision_score(y_pred, labels, average=None))
    print("Recall: ", recall_score(y_pred, labels, average=None))
    print()


if __name__ == "__main__":
    dataset_list = ["syn1", "syn4", "syn5"]
    dataset_dict = {}

    for dataset in dataset_list:

        with open("../data/gnn_explainer/{}.pickle".format(dataset), "rb") as f:
            dataset_dict[dataset] = pickle.load(f)

    for k, v in dataset_dict.items():
        evaluate_model(v, k)
