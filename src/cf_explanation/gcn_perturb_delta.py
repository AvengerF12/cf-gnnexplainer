import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils.utils import get_degree_matrix, BernoulliMLSample, create_symm_matrix_tril


class GCNSyntheticPerturbDelta(nn.Module):
    """
    3-layer GCN used in GNN Explainer synthetic tasks
    """
    def __init__(self, model, nclass, adj, num_nodes, alpha, beta, gamma, task,
                 edge_del=False, edge_add=False, bernoulli=False, rand_init=0.5,
                 cem_mode=None, device=None):
        super(GCNSyntheticPerturbDelta, self).__init__()
        self.model = model
        # The adj mat is stored since each instance of the explainer deals with a single node
        self.adj = adj
        self.nclass = nclass
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Used to find the appropriate part of P to perturbate (w/o padding) and compute flattened
        # layer for graph classification (differs from num_nodes_adj only when task = graph-class)
        self.num_nodes_actual = num_nodes

        self.bernoulli = bernoulli
        self.rand_init = rand_init
        self.cem_mode = cem_mode
        self.device = device

        self.BML = BernoulliMLSample.apply

        self.edge_del = edge_del  # Can the explainer delete new edges to the graph
        self.edge_add = edge_add  # Can the explainer add new edges to the graph

        if not edge_del and not edge_add:
            raise RuntimeError("GCNSyntheticPerturbDelta: need to specify allowed add/del op")

        # Number of nodes in the adj, in case of graph-class includes padding
        self.num_nodes_adj = self.adj.shape[1]
        self.init_eps = 10**-6

        if 0 < self.rand_init < self.init_eps:
            raise RuntimeError("GCNSyntheticPerturbDelta: rand_init value too small")

        # The optimizer will affect only the elements below the diag of this matrix
        # This is enforced through the function create_symm_matrix_tril(), which constructs the 
        # symmetric matrix to optimize using only the lower triangular elements of P_tril
        # Note: no diagonal, it is assumed to be always 0/no self-connections allowed
        if self.bernoulli:
            self.P_tril = Parameter(torch.zeros((self.num_nodes_actual, self.num_nodes_actual),
                                                device=self.device))
        else:
            # Need to guarantee that the initial permutation matrix is all 0s after applying sigm
            self.P_tril = Parameter(torch.full((self.num_nodes_actual, self.num_nodes_actual),
                                               -self.init_eps, device=self.device))

        # The idea behind the init is simply to break any symmetries in the parameters, allowing
        # for more diverse explanations by avoiding the simultaneous addition/deletion of relevant
        # edges
        if self.rand_init > 0:
            if self.bernoulli:
                torch.nn.init.uniform_(self.P_tril, a=0.5-self.rand_init, b=0.5-self.init_eps)
            else:
                torch.nn.init.uniform_(self.P_tril, a=-self.rand_init, b=-self.init_eps)


    def forward(self, x):

        diff_act, pred_act = None, None

        if self.bernoulli:
            diff_act, pred_act = self.__forward_bernoulli(x)
        else:
            diff_act, pred_act = self.__forward_std(x)

        return diff_act, pred_act


    def __forward_std(self, x):
        # Use sigmoid to bound P_hat in [0,1]
        P_hat_symm = torch.sigmoid(self.P_tril)
        P_hat_symm = create_symm_matrix_tril(P_hat_symm, self.num_nodes_adj)
        P = (P_hat_symm >= 0.5).float()  # Threshold P_hat

        # Note: identity matrix is added in normalize_adj() inside model
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

        output_diff = self.model(x, A_tilde_diff.expand(1, -1, -1)).squeeze()
        output_pred = self.model(x, A_tilde_pred.expand(1, -1, -1)).squeeze()

        return output_diff, output_pred


    def __forward_bernoulli(self, x):

        P_hat_symm = create_symm_matrix_tril(self.P_tril, self.num_nodes_adj)
        P = self.BML(P_hat_symm)  # Threshold P_hat
        delta = 0

        # Note: identity matrix is added in normalize_adj() inside model
        if self.edge_add:
            delta += (1 - self.adj) * P

        if self.edge_del:
            delta -= P * self.adj

        A_tilde = self.adj + delta

        output = self.model(x, A_tilde.expand(1, -1, -1)).squeeze()

        return output, output


    def loss(self, output, y_pred_orig, y_pred_new_actual, prev_expls):

        if self.bernoulli:

            if self.cem_mode is None or self.cem_mode == "PN":
                res = self.__loss_bernoulli(output, y_pred_orig, y_pred_new_actual, prev_expls)
            elif self.cem_mode == "PP":
                res = self.__loss_PP_bernoulli(output, y_pred_orig, y_pred_new_actual, prev_expls)

        else:

            if self.cem_mode is None or self.cem_mode == "PN":
                res = self.__loss_std(output, y_pred_orig, y_pred_new_actual, prev_expls)
            elif self.cem_mode == "PP":
                res = self.__loss_PP_std(output, y_pred_orig, y_pred_new_actual, prev_expls)

        return res


    def __loss_std(self, output, y_pred_orig, y_pred_new_actual, prev_expls):
        P_hat_symm = torch.sigmoid(self.P_tril)
        P_hat_symm = create_symm_matrix_tril(P_hat_symm, self.num_nodes_adj)
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

        cf_adj_diff = self.adj + delta_diff
        cf_adj_actual = self.adj + delta_actual

        # Want negative in front to maximize loss instead of minimizing it to find CFs
        loss_pred = - F.nll_loss(output, y_pred_orig)
        # Number of edges changed (symmetrical), used for the metrics
        loss_graph_dist_actual = torch.sum(torch.abs(delta_actual)) / 2
        # Relaxation to continuous space of loss_graph_dist_actual, used for the loss
        loss_graph_dist_diff = torch.sum(torch.abs(delta_diff)) / 2

        diversity_loss = 0
        for expl in prev_expls:
            diversity_loss += F.cross_entropy(cf_adj_diff, expl)

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = self.alpha * pred_same * loss_pred + self.beta * loss_graph_dist_diff \
            - self.gamma * diversity_loss

        return loss_total, loss_graph_dist_actual, cf_adj_diff, cf_adj_actual


    def __loss_bernoulli(self, output, y_pred_orig, y_pred_new_actual, prev_expls):
        P_hat_symm = create_symm_matrix_tril(self.P_tril, self.num_nodes_adj)
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

        diversity_loss = 0
        for expl in prev_expls:
            diversity_loss += F.cross_entropy(cf_adj, expl)

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = self.alpha * pred_same * loss_pred + self.beta * loss_graph_dist \
            - self.gamma * diversity_loss

        return loss_total, loss_graph_dist, cf_adj, cf_adj


    def __loss_PP_std(self, output, y_pred_orig, y_pred_new_actual, prev_expls):
        P_hat_symm = torch.sigmoid(self.P_tril)
        P_hat_symm = create_symm_matrix_tril(P_hat_symm, self.num_nodes_adj)
        P = (P_hat_symm >= 0.5).float()  # Threshold P_hat

        # Note: flipped the boolean since we want the same prediction
        pred_diff = (y_pred_new_actual != y_pred_orig).float()

        # Init to 0, since it will be broadcasted into the appropriate shape by torch
        delta_diff = 0
        delta_actual = 0

        # edge_del equivalent
        delta_diff -= P_hat_symm * self.adj
        delta_actual -= P * self.adj

        cf_adj_diff = self.adj + delta_diff
        cf_adj_actual = self.adj + delta_actual

        # Note: the negative sign is gone since we want to keep the same prediction
        loss_pred = F.nll_loss(output, y_pred_orig)
        # Number of edges in neighbourhood (symmetrical)
        # Note: here we are interested in finding the most sparse cf_adj with the same pred
        loss_graph_dist = torch.sum(torch.abs(cf_adj_diff)) / 2
        # Note: in order to generate the best PP we need to minimize the number of entries in the
        # cf_adj, however in order to get a better understanding the number of edges deleted is 
        # more useful to the end user. Therefore this number is the one saved inside each expl
        # generated
        loss_graph_dist_actual = torch.sum(torch.abs(self.adj - cf_adj_actual)) / 2

        diversity_loss = 0
        for expl in prev_expls:
            diversity_loss += F.cross_entropy(cf_adj_diff, expl)

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = self.alpha * pred_diff * loss_pred + self.beta * loss_graph_dist \
            - self.gamma * diversity_loss

        return loss_total, loss_graph_dist_actual, cf_adj_diff, cf_adj_actual


    def __loss_PP_bernoulli(self, output, y_pred_orig, y_pred_new_actual, prev_expls):
        P_hat_symm = create_symm_matrix_tril(self.P_tril, self.num_nodes_adj)
        P = self.BML(P_hat_symm)  # Threshold P_hat

        # Note: flipped the boolean since we want the same prediction
        pred_diff = (y_pred_new_actual != y_pred_orig).float()
        delta = 0

        # edge_del equivalent
        delta -= P * self.adj
        cf_adj = self.adj + delta

        # Note: the negative sign is gone since we want to keep the same prediction
        loss_pred = F.nll_loss(output, y_pred_orig)
        # Number of edges in neighbourhood (symmetrical)
        # Note: here we are interested in finding the most sparse cf_adj with the same pred
        loss_graph_dist = torch.sum(torch.abs(cf_adj)) / 2
        # Note: in order to generate the best PP we need to minimize the number of entries in the
        # cf_adj, however in order to get a better understanding the number of edges deleted is 
        # more useful to the end user. Therefore this number is the one saved inside each cf_example
        # generated
        loss_graph_dist_actual = torch.sum(torch.abs(self.adj - cf_adj)) / 2

        diversity_loss = 0
        for expl in prev_expls:
            diversity_loss += F.cross_entropy(cf_adj, expl)

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = self.alpha * pred_diff * loss_pred + self.beta * loss_graph_dist \
            - self.gamma * diversity_loss

        return loss_total, loss_graph_dist_actual, cf_adj, cf_adj
