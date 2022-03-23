import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils.utils import get_degree_matrix, normalize_adj, BernoulliMLSample, create_symm_matrix_tril
from gcn import GraphConvolution, GCNSynthetic


class GCNSyntheticPerturbDelta(nn.Module):
    """
    3-layer GCN used in GNN Explainer synthetic tasks
    """
    def __init__(self, nfeat, nhid, nout, nclass, adj, dropout, beta, edge_del=False,
                 edge_add=False, bernoulli=False, device=None):
        super(GCNSyntheticPerturbDelta, self).__init__()
        # The adj mat is stored since each instance of the explainer deals with a single node
        self.adj = adj
        self.nclass = nclass
        self.beta = beta
        self.num_nodes = self.adj.shape[0]
        self.bernoulli = bernoulli
        self.device = device

        self.BML = BernoulliMLSample.apply

        self.edge_del = edge_del  # Can the model delete new edges to the graph
        self.edge_add = edge_add  # Can the model add new edges to the graph

        if not edge_del and not edge_add:
            raise RuntimeError("GCNSyntheticPerturbDelta: need to specify allowed add/del op")

        # The optimizer will affect only the elements below the diag of this matrix
        # This is enforced through the function create_symm_matrix_tril(), which construct the 
        # symmetric matrix to optimize using only the lower triangular elements of P_tril
        # Note: no diagonal, it is assumed to be always 0/no self-connections allowed
        self.P_tril = Parameter(torch.FloatTensor(torch.zeros(self.num_nodes, self.num_nodes)))

        # Avoid creating an eye matrix for each normalize_adj op, re-use the same one
        self.norm_eye = torch.eye(self.num_nodes, device=device)

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nout)
        self.lin = nn.Linear(nhid + nhid + nout, nclass)
        self.dropout = dropout


    def __apply_model(self, x, norm_adj):

        x1 = F.relu(self.gc1(x, norm_adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, norm_adj))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(x2, norm_adj)
        x = self.lin(torch.cat((x1, x2, x3), dim=1))

        return x


    def forward(self, x):

        diff_act, pred_act = None, None

        if self.bernoulli:
            diff_act, pred_act = self.__forward_bernoulli(x)
        else:
            diff_act, pred_act = self.__forward_std(x)

        return diff_act, pred_act


    def __forward_std(self, x):
        # Use sigmoid to bound P_hat in [0,1]
        # Applying sigmoid on P_tril instead of P_hat_symm avoids problems with
        # diagonal equal to 1 when using edge_add, since sigmoid(0)=0.5
        P_hat_symm = torch.sigmoid(self.P_tril)
        P_hat_symm = create_symm_matrix_tril(P_hat_symm, self.device)
        P = (P_hat_symm >= 0.5).float()  # Threshold P_hat

        # Init A_tilde* before applying allowed operations
        # Note: identity matrix is added in normalize_adj()
        delta_diff = 0
        delta_pred = 0

        # TODO: check for numerical errors
        if self.edge_add:
            delta_diff += (1 - self.adj.int()) * P_hat_symm
            delta_pred += (1 - self.adj.int()) * P

        if self.edge_del:
            delta_diff -= P_hat_symm * self.adj.int()
            delta_pred -= P * self.adj.int()

        A_tilde_diff = self.adj + delta_diff
        A_tilde_pred = self.adj + delta_pred

        norm_adj_diff = normalize_adj(A_tilde_diff, self.norm_eye, self.device)
        norm_adj_pred = normalize_adj(A_tilde_pred, self.norm_eye, self.device)

        output_diff = self.__apply_model(x, norm_adj_diff)
        output_pred = self.__apply_model(x, norm_adj_pred)

        return F.log_softmax(output_diff, dim=1), F.log_softmax(output_pred, dim=1)


    def __forward_bernoulli(self, x):

        P_hat_symm = create_symm_matrix_tril(self.P_tril, self.device)
        P = self.BML(P_hat_symm)  # Threshold P_hat
        delta = 0

        # Note: identity matrix is added in normalize_adj()
        if self.edge_add:
            delta += (1 - self.adj.int()) * P

        if self.edge_del:
            delta -= P * self.adj.int()

        A_tilde = self.adj + delta

        norm_adj = normalize_adj(A_tilde, self.norm_eye, self.device)

        output = self.__apply_model(x, norm_adj)
        act_output = F.log_softmax(output, dim=1)

        return act_output, act_output


    def loss_std(self, output, y_pred_orig, y_pred_new_actual):
        P_hat_symm = torch.sigmoid(self.P_tril)
        P_hat_symm = create_symm_matrix_tril(P_hat_symm, self.device)
        P = (P_hat_symm >= 0.5).float()  # Threshold P_hat

        pred_same = (y_pred_new_actual == y_pred_orig).float()

        # Init to 0, since it will be broadcasted into the appropriate shape by torch
        delta_diff = 0
        delta_actual = 0

        if self.edge_add:
            mat_constr = 1 - self.adj.int()
            delta_diff += mat_constr * P_hat_symm
            delta_actual += mat_constr * P

        if self.edge_del:
            delta_diff -= P_hat_symm * self.adj.int()
            delta_actual -= P * self.adj

        cf_adj_actual = self.adj + delta_actual

        # Want negative in front to maximize loss instead of minimizing it to find CFs
        loss_pred = - F.nll_loss(output, y_pred_orig)
        # Number of edges changed (symmetrical), used for the metrics
        loss_graph_dist_actual = torch.sum(torch.abs(delta_actual)) / 2
        # Relaxation to continuous space of loss_graph_dist_actual, used for the loss
        loss_graph_dist_diff = torch.sum(torch.abs(delta_diff)) / 2

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist_diff

        return loss_total, loss_pred, loss_graph_dist_actual, cf_adj_actual


    # TODO: try bi-modal regulariser
    def loss_bernoulli(self, output, y_pred_orig, y_pred_new_actual):
        P_hat_symm = create_symm_matrix_tril(self.P_tril, self.device)
        P = self.BML(P_hat_symm)  # Threshold P_hat

        pred_same = (y_pred_new_actual == y_pred_orig).float()
        delta = 0

        # Note: the differentiable and actual formulations are identical
        if self.edge_add:
            delta += (1 - self.adj.int()) * P

        if self.edge_del:
            delta -= P * self.adj.int()

        cf_adj = self.adj + delta

        # Want negative in front to maximize loss instead of minimizing it to find CFs
        loss_pred = - F.nll_loss(output, y_pred_orig)
        # Number of edges changed (symmetrical)
        loss_graph_dist = torch.sum(torch.abs(delta)) / 2

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist

        return loss_total, loss_pred, loss_graph_dist, cf_adj
