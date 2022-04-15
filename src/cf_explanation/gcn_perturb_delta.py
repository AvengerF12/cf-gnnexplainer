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
    def __init__(self, model, nclass, adj, num_nodes, beta, task,
                 edge_del=False, edge_add=False, bernoulli=False, rand_init=True, device=None):
        super(GCNSyntheticPerturbDelta, self).__init__()
        self.model = model
        # The adj mat is stored since each instance of the explainer deals with a single node
        self.adj = adj
        self.nclass = nclass
        self.beta = beta
        self.task = task

        # Used to find the appropriate part of P to perturbate (w/o padding) and compute flattened
        # layer for graph classification (differs from num_nodes_adj only when task = graph-class)
        self.num_nodes_actual = num_nodes

        self.bernoulli = bernoulli
        self.device = device

        self.BML = BernoulliMLSample.apply

        self.edge_del = edge_del  # Can the explainer delete new edges to the graph
        self.edge_add = edge_add  # Can the explainer add new edges to the graph

        if not edge_del and not edge_add:
            raise RuntimeError("GCNSyntheticPerturbDelta: need to specify allowed add/del op")

        allowed_tasks = ["node-class", "graph-class"]
        if self.task not in allowed_tasks:
            raise RuntimeError("GCNSynthetic: invalid task specified")

        # Number of nodes in the adj, in case of graph-class includes padding
        self.num_nodes_adj = self.adj.shape[0]

        # The optimizer will affect only the elements below the diag of this matrix
        # This is enforced through the function create_symm_matrix_tril(), which constructs the 
        # symmetric matrix to optimize using only the lower triangular elements of P_tril
        # Note: no diagonal, it is assumed to be always 0/no self-connections allowed
        self.P_tril = Parameter(torch.zeros((self.num_nodes_actual, self.num_nodes_actual),
                                            device=self.device))

        # The idea behind the init is simply to break any symmetries in the parameters, allowing
        # for more diverse explanations by avoiding the simultaneous addition/deletion of relevant
        # edges
        if rand_init:
            if self.bernoulli:
                torch.nn.init.uniform_(self.P_tril, a=0, b=0.4)
            else:
                torch.nn.init.uniform_(self.P_tril, a=-0.4, b=0)

        # Avoid creating an eye matrix for each normalize_adj op, re-use the same one
        self.norm_eye = torch.eye(self.num_nodes_adj, device=device)


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
        # diagonal equal to 1 when using edge_add, since sigmoid(0)=0.5 >= threshold already
        P_hat_symm = torch.sigmoid(self.P_tril)
        P_hat_symm = create_symm_matrix_tril(P_hat_symm, self.num_nodes_adj, self.device)
        P = (P_hat_symm >= 0.5).float()  # Threshold P_hat

        # Note: identity matrix is added in normalize_adj()
        delta_diff = 0
        delta_pred = 0

        if self.edge_add:
            delta_diff += (1 - self.adj) * P_hat_symm
            delta_pred += (1 - self.adj) * P

        if self.edge_del:
            delta_diff -= P_hat_symm * self.adj
            delta_pred -= P * self.adj

        A_tilde_diff = self.adj + delta_diff
        A_tilde_pred = self.adj + delta_pred

        norm_adj_diff = normalize_adj(A_tilde_diff, self.norm_eye, self.device)
        norm_adj_pred = normalize_adj(A_tilde_pred, self.norm_eye, self.device)

        output_diff = self.model(x, norm_adj_diff)
        output_pred = self.model(x, norm_adj_pred)

        return output_diff, output_pred


    def __forward_bernoulli(self, x):

        P_hat_symm = create_symm_matrix_tril(self.P_tril, self.num_nodes_adj, self.device)
        P = self.BML(P_hat_symm)  # Threshold P_hat
        delta = 0

        # Note: identity matrix is added in normalize_adj()
        if self.edge_add:
            delta += (1 - self.adj) * P

        if self.edge_del:
            delta -= P * self.adj

        A_tilde = self.adj + delta

        norm_adj = normalize_adj(A_tilde, self.norm_eye, self.device)

        output = self.model(x, norm_adj)

        return output, output


    def loss_std(self, output, y_pred_orig, y_pred_new_actual):
        P_hat_symm = torch.sigmoid(self.P_tril)
        P_hat_symm = create_symm_matrix_tril(P_hat_symm, self.num_nodes_adj, self.device)
        P = (P_hat_symm >= 0.5).float()  # Threshold P_hat

        pred_same = (y_pred_new_actual == y_pred_orig).float()

        # Init to 0, since it will be broadcasted into the appropriate shape by torch
        delta_diff = 0
        delta_actual = 0

        if self.edge_add:
            mat_constr = 1 - self.adj
            delta_diff += mat_constr * P_hat_symm
            delta_actual += mat_constr * P

        if self.edge_del:
            delta_diff -= P_hat_symm * self.adj
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


    def loss_bernoulli(self, output, y_pred_orig, y_pred_new_actual):
        P_hat_symm = create_symm_matrix_tril(self.P_tril, self.num_nodes_adj, self.device)
        P = self.BML(P_hat_symm)  # Threshold P_hat

        pred_same = (y_pred_new_actual == y_pred_orig).float()
        delta = 0

        # Note: the differentiable and actual formulations are identical
        if self.edge_add:
            delta += (1 - self.adj) * P

        if self.edge_del:
            delta -= P * self.adj

        cf_adj = self.adj + delta

        # Want negative in front to maximize loss instead of minimizing it to find CFs
        loss_pred = - F.nll_loss(output, y_pred_orig)
        # Number of edges changed (symmetrical)
        loss_graph_dist = torch.sum(torch.abs(delta)) / 2

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist

        return loss_total, loss_pred, loss_graph_dist, cf_adj
