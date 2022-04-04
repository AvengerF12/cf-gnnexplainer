import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils.utils import get_degree_matrix, normalize_adj, BernoulliMLSample, create_symm_matrix_tril
from gcn import GraphConvolution, GCNSynthetic


class GCNSyntheticPerturbOrig(nn.Module):
    """
    3-layer GCN used in GNN Explainer synthetic tasks
    """
    def __init__(self, nfeat, nhid, nout, nclass, adj, dropout, beta, task, edge_del=False,
                 edge_add=False, bernoulli=False, device=None):
        super(GCNSyntheticPerturbOrig, self).__init__()
        # The adj mat is stored since each instance of the explainer deals with a single node
        self.adj = adj
        self.nclass = nclass
        self.beta = beta
        self.num_nodes = self.adj.shape[0]
        self.task = task
        self.bernoulli = bernoulli
        self.device = device

        self.BML = BernoulliMLSample.apply

        self.edge_del = edge_del  # Can the model delete new edges to the graph
        self.edge_add = edge_add  # Can the model add new edges to the graph

        if not edge_del and not edge_add:
            raise RuntimeError("GCNSyntheticPerturbOrig: need to specify allowed add/del op")
        elif edge_del and edge_add:
            print("Note: in this implementation enabling edge_add allows for both add and del")

        allowed_tasks = ["node-class", "graph-class"]
        if self.task not in allowed_tasks:
            raise RuntimeError("GCNSynthetic: invalid task specified")

        # The optimizer will affect only the elements below the diag of this matrix
        # This is enforced through the function create_symm_matrix_tril(), which construct the 
        # symmetric matrix to optimize using only the lower triangular elements of P_tril
        # Note: no diagonal, it is assumed to be always 0/no self-connections allowed
        if self.edge_add:
            # Initialize the matrix to the lower triangular part of the adj
            self.P_tril = Parameter(torch.tril(self.adj, -1).detach())
        else:
            self.P_tril = Parameter(torch.FloatTensor(torch.ones(self.num_nodes, self.num_nodes)))

        # Avoid creating an eye matrix for each normalize_adj op, re-use the same one
        self.norm_eye = torch.eye(self.num_nodes, device=device)

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nout)

        if self.task == "graph-class":
            self.dim_lin = (nhid + nhid + nout) * self.num_nodes
            self.lin = nn.Linear(self.dim_lin, nclass)
        elif self.task == "node-class":
            self.dim_lin = nhid + nhid + nout
            self.lin = nn.Linear(self.dim_lin, nclass)

        self.dropout = dropout


    def __apply_model(self, x, norm_adj):

        x1 = F.relu(self.gc1(x, norm_adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, norm_adj))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(x2, norm_adj)

        if self.task == "graph-class":
            lin_in = torch.flatten(torch.cat((x1, x2, x3), dim=1))
        elif self.task == "node-class":
            lin_in = torch.cat((x1, x2, x3), dim=1)

        x = self.lin(lin_in)

        if self.task == "graph-class":
            softmax_out = F.log_softmax(x, dim=0)
        elif self.task == "node-class":
            softmax_out = F.log_softmax(x, dim=1)

        return softmax_out


    def forward(self, x):

        diff_act, pred_act = None, None

        if self.bernoulli:
            diff_act, pred_act = self.__forward_bernoulli(x)
        else:
            diff_act, pred_act = self.__forward_std(x)

        return diff_act, pred_act


    def __forward_std(self, x):
        # Applying sigmoid on P_tril instead of P_hat_symm avoids problems with
        # diagonal equal to 1 when using edge_add, since sigmoid(0)=0.5
        P_hat_symm = torch.sigmoid(self.P_tril)
        P_hat_symm = create_symm_matrix_tril(P_hat_symm, self.device)
        P = (P_hat_symm >= 0.5).float()  # Threshold P_hat

        # Note: identity matrix is added in normalize_adj()
        if self.edge_add:  # Learn new adj matrix directly
            # Use sigmoid to bound P_hat in [0,1]
            A_tilde_diff = P_hat_symm
            A_tilde_pred = P
        else:  # Learn only P_hat => only edge deletions
            A_tilde_diff = P_hat_symm * self.adj
            A_tilde_pred = P * self.adj

        norm_adj_diff = normalize_adj(A_tilde_diff, self.norm_eye, self.device)
        norm_adj_pred = normalize_adj(A_tilde_pred, self.norm_eye, self.device)

        output_diff = self.__apply_model(x, norm_adj_diff)
        output_pred = self.__apply_model(x, norm_adj_pred)

        return output_diff, output_pred


    def __forward_bernoulli(self, x):

        P_hat_symm = create_symm_matrix_tril(self.P_tril, self.device)
        P = self.BML(P_hat_symm)  # Threshold P_hat

        # Note: identity matrix is added in normalize_adj()
        if self.edge_add:  # Learn new adj matrix directly
            A_tilde = P
        else:       # Learn only P_hat => only edge deletions
            A_tilde = P * self.adj

        norm_adj = normalize_adj(A_tilde, self.norm_eye, self.device)

        output = self.__apply_model(x, norm_adj)

        return output, output


    def loss_std(self, output, y_pred_orig, y_pred_new_actual):
        P_hat_symm = torch.sigmoid(self.P_tril)
        P_hat_symm = create_symm_matrix_tril(P_hat_symm, self.device)
        P = (P_hat_symm >= 0.5).float()  # Threshold P_hat

        pred_same = (y_pred_new_actual == y_pred_orig).float()

        if self.edge_add:
            cf_adj = P_hat_symm
            cf_adj_actual = P
        else:
            cf_adj = P_hat_symm * self.adj
            cf_adj_actual = P * self.adj

        # Want negative in front to maximize loss instead of minimizing it to find CFs
        loss_pred = - F.nll_loss(output, y_pred_orig)
        # Number of edges changed (symmetrical), used for the metrics
        loss_graph_dist_actual = torch.sum(torch.abs(cf_adj_actual - self.adj)) / 2
        # Relaxation to continuous space of loss_graph_dist_actual, used for the loss
        loss_graph_dist = torch.sum(torch.abs(cf_adj - self.adj)) / 2

        # Zero-out loss_pred with pred_same if prediction flips
        # Note: the distance loss is non-differentiable => it's not optimized directly.
        # It only comes into play when comparing the current loss with best loss in cf_explainer
        # The results obtained using the hyperparameters on the original paper are identical
        # w/wo the dist loss.
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist

        return loss_total, loss_pred, loss_graph_dist_actual, cf_adj_actual


    # TODO: try bi-modal regulariser
    def loss_bernoulli(self, output, y_pred_orig, y_pred_new_actual):
        P_hat_symm = create_symm_matrix_tril(self.P_tril, self.device)
        P = self.BML(P_hat_symm)  # Threshold P_hat

        pred_same = (y_pred_new_actual == y_pred_orig).float()

        # Note: the differentiable and actual formulations are identical
        if self.edge_add:
            cf_adj = P
        else:
            cf_adj = P * self.adj

        # Want negative in front to maximize loss instead of minimizing it to find CFs
        loss_pred = - F.nll_loss(output, y_pred_orig)
        # Number of edges changed (symmetrical)
        loss_graph_dist = torch.sum(torch.abs(cf_adj - self.adj)) / 2

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist

        return loss_total, loss_pred, loss_graph_dist, cf_adj
