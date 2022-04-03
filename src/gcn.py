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
    3-layer GCN used in GNN Explainer synthetic tasks, including
    """
    def __init__(self, nfeat, nhid, nout, nclass, dropout, graph_class=False, num_nodes=None):
        super(GCNSynthetic, self).__init__()

        self.graph_class = graph_class

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nout)

        if self.graph_class:
            self.dim_lin = (nhid + nhid + nout) * num_nodes
            self.lin = nn.Linear(self.dim_lin, nclass)
        else:
            self.dim_lin = nhid + nhid + nout
            self.lin = nn.Linear(self.dim_lin, nclass)

        self.dropout = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, adj))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(x2, adj)

        if self.graph_class:
            lin_in = torch.flatten(torch.cat((x1, x2, x3), dim=1))
        else:
            lin_in = torch.cat((x1, x2, x3), dim=1)

        x = self.lin(lin_in)

        if self.graph_class:
            softmax = F.log_softmax(x, dim=0)
        else:
            softmax = F.log_softmax(x, dim=1)

        return softmax

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
