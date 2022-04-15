# Based on https://github.com/tkipf/pygcn/blob/master/pygcn/

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCNSynthetic(nn.Module):
    """
    3-layer GCN used in GNN Explainer synthetic tasks
    """
    # num_nodes is used to determine the size of the flattened layer
    # It assumes that the max number of nodes in the graph adj matrices is num_nodes
    def __init__(self, nfeat, nhid, nout, nclass, dropout):
        super(GCNSynthetic, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nout)

        self.dim_lin = nhid + nhid + nout
        self.lin = nn.Linear(self.dim_lin, nclass)

        self.dropout = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, adj))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(x2, adj)

        lin_in = torch.cat((x1, x2, x3), dim=1)
        x = self.lin(lin_in)
        softmax_out = F.log_softmax(x, dim=1)

        return softmax_out

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class GraphAttNet(nn.Module):
    """
    3-layer Graph Attention Network used in GNN Explainer graph classification tasks
    """
    # num_nodes is used to determine the size of the flattened layer
    # It assumes that the max number of nodes in the graph adj matrices is num_nodes
    def __init__(self, nfeat, nhid, nout, nclass, dropout):
        super(GraphAttNet, self).__init__()

        self.n_layers = 3

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nout)

        self.lin_dim = nhid * (self.n_layers - 1) + nout
        self.lin = nn.Linear(self.lin_dim, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        # Note: needs dim=1 in case of mini-batch training (also a big code refactor)
        out_list = []

        x1 = F.relu(self.gc1(x, adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        out, _ = torch.max(x1, dim=0)
        out_list.append(out)

        x2 = F.relu(self.gc2(x1, adj))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        out, _ = torch.max(x2, dim=0)
        out_list.append(out)

        x3 = self.gc3(x2, adj)
        out, _ = torch.max(x3, dim=0)
        out_list.append(out)

        lin_in = torch.cat(out_list, dim=0)
        x = self.lin(lin_in)
        softmax_out = F.log_softmax(x, dim=0)

        return softmax_out

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
