import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils.utils import get_degree_matrix, BernoulliMLSample, create_symm_matrix_tril


class GCNSyntheticPerturbOrig(nn.Module):
    """
    3-layer GCN used in GNN Explainer synthetic tasks
    """
    def __init__(self, model, nclass, adj, num_nodes, alpha, beta, gamma, task,
                 edge_del=False, edge_add=False, bernoulli=False, rand_init=True,
                 cem_mode=None, device=None):
        super(GCNSyntheticPerturbOrig, self).__init__()
        self.model = model
        # The adj mat is stored since each instance of the explainer deals with a single node
        self.adj = adj
        self.nclass = nclass
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bernoulli = bernoulli

        # Used to find the appropriate part of P to perturbate (w/o padding) and compute flattened
        # layer for graph classification (differs from num_nodes_adj only when task = graph-class)
        self.num_nodes_actual = num_nodes

        self.cem_mode = cem_mode
        self.device = device

        self.BML = BernoulliMLSample.apply

        self.edge_del = edge_del  # Can the model delete new edges to the graph
        self.edge_add = edge_add  # Can the model add new edges to the graph

        if not edge_del and not edge_add:
            raise RuntimeError("GCNSyntheticPerturbOrig: need to specify allowed add/del op")
        elif edge_del and edge_add:
            print("Note: in this implementation enabling edge_add allows for both add and del")

        # Number of nodes in the adj, in case of graph-class includes padding
        self.num_nodes_adj = self.adj.shape[1]

       # The optimizer will affect only the elements below the diag of this matrix
       # Note: no diagonal, it is assumed to be always 0/no self-connections allowed
        if self.edge_add:
            # Initialize the matrix to the lower triangular part of the adj
            self.P_tril = Parameter(torch.tril(self.adj, -1).detach())
        else:
            self.P_tril = Parameter(torch.FloatTensor(torch.ones(self.num_nodes_actual,
                                                                 self.num_nodes_actual)))

        # The idea behind the init is simply to break any symmetries in the parameters, allowing
        # for more diverse explanations by avoiding the simultaneous addition/deletion of relevant
        # edges
        if rand_init and self.edge_add:
            if self.bernoulli:
                torch.nn.init.uniform_(self.P_tril[self.P_tril == 1], a=0.6, b=1)
                torch.nn.init.uniform_(self.P_tril[self.P_tril == 0], a=0, b=0.4)
            else:
                torch.nn.init.uniform_(self.P_tril[self.P_tril == 1], a=1, b=1.4)
                # Note: the value must be below 0, otherwise the sigmoid will be >= 0.5
                torch.nn.init.uniform_(self.P_tril[self.P_tril == 0], a=-0.5, b=-0.1)
        elif rand_init:
            if self.bernoulli:
                torch.nn.init.uniform_(self.P_tril, a=0.6, b=1)
            else:
                torch.nn.init.uniform_(self.P_tril, a=0.1, b=0.5)


    def forward(self, x):

        diff_act, pred_act = None, None

        if self.bernoulli:
            diff_act, pred_act = self.__forward_bernoulli(x)
        else:
            diff_act, pred_act = self.__forward_std(x)

        return diff_act, pred_act


    def __forward_std(self, x):
        # Applying sigmoid on P_tril instead of P_hat_symm avoids problems with
        # diagonal equal to 1 during training when using edge_add, since sigmoid(0)=0.5
        P_hat_symm = torch.sigmoid(self.P_tril)
        P_hat_symm = create_symm_matrix_tril(P_hat_symm, self.num_nodes_adj)
        P = (P_hat_symm >= 0.5).float()  # Threshold P_hat

        # Note: identity matrix is added in normalize_adj() inside the model
        if self.edge_add:  # Learn new adj matrix directly
            # Use sigmoid to bound P_hat in [0,1]
            A_tilde_diff = P_hat_symm
            A_tilde_pred = P
        else:  # Learn only P_hat => only edge deletions
            A_tilde_diff = P_hat_symm * self.adj
            A_tilde_pred = P * self.adj

        output_diff = self.model(x, A_tilde_diff.expand(1, -1, -1)).squeeze()
        output_pred = self.model(x, A_tilde_pred.expand(1, -1, -1)).squeeze()

        return output_diff, output_pred


    def __forward_bernoulli(self, x):

        P_hat_symm = create_symm_matrix_tril(self.P_tril, self.num_nodes_adj)
        P = self.BML(P_hat_symm)  # Threshold P_hat

        # Note: identity matrix is added in normalize_adj() inside model
        if self.edge_add:  # Learn new adj matrix directly
            A_tilde = P
        else:       # Learn only P_hat => only edge deletions
            A_tilde = P * self.adj

        output = self.model(x, A_tilde.expand(1, -1, -1)).squeeze()

        return output, output


    def loss(self, output, y_pred_orig, y_pred_new_actual, prev_expls):

        if self.bernoulli:

            if self.cem_mode is None:
                res = self.__loss_bernoulli(output, y_pred_orig, y_pred_new_actual, prev_expls)
            elif self.cem_mode == "PP":
                res = self.__loss_PP_bernoulli(output, y_pred_orig, y_pred_new_actual, prev_expls)
            elif self.cem_mode == "PN":
                raise RuntimeError("GCNSyntheticPerturbOrig: PN is not implemented")

        else:

            if self.cem_mode is None:
                res = self.__loss_std(output, y_pred_orig, y_pred_new_actual, prev_expls)
            elif self.cem_mode == "PP":
                res = self.__loss_PP_std(output, y_pred_orig, y_pred_new_actual, prev_expls)
            elif self.cem_mode == "PN":
                raise RuntimeError("GCNSyntheticPerturbOrig: PN is not implemented")

        return res

    def __loss_std(self, output, y_pred_orig, y_pred_new_actual, prev_expls):
        P_hat_symm = torch.sigmoid(self.P_tril)
        P_hat_symm = create_symm_matrix_tril(P_hat_symm, self.num_nodes_adj)
        P = (P_hat_symm >= 0.5).float()  # Threshold P_hat

        pred_same = (y_pred_new_actual == y_pred_orig).float()

        if self.edge_add:
            cf_adj_diff = P_hat_symm
            cf_adj_actual = P
        else:
            cf_adj_diff = P_hat_symm * self.adj
            cf_adj_actual = P * self.adj

        # Want negative in front to maximize loss instead of minimizing it to find CFs
        loss_pred = - F.nll_loss(output, y_pred_orig)
        # Number of edges changed (symmetrical), used for the metrics
        loss_graph_dist_actual = torch.sum(torch.abs(cf_adj_actual - self.adj)) / 2
        # Relaxation to continuous space of loss_graph_dist_actual, used for the loss
        loss_graph_dist = torch.sum(torch.abs(cf_adj_diff - self.adj)) / 2

        diversity_loss = 0
        for expl in prev_expls:
            diversity_loss += F.cross_entropy(cf_adj_diff, expl)

        # Zero-out loss_pred with pred_same if prediction flips
        # Note: the distance loss in the original paper is non-differentiable => 
        #   it's not optimized directly.
        # It only comes into play when comparing the current loss with best loss in cf_explainer
        # The results obtained using the hyperparameters on the original paper are identical
        # w/wo the dist loss.
        loss_total = self.alpha * pred_same * loss_pred + self.beta * loss_graph_dist \
            - self.gamma * diversity_loss

        return loss_total, loss_graph_dist_actual, cf_adj_diff, cf_adj_actual


    def __loss_bernoulli(self, output, y_pred_orig, y_pred_new_actual, prev_expls):
        P_hat_symm = create_symm_matrix_tril(self.P_tril, self.num_nodes_adj)
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

        # edge_del equivalent
        cf_adj_diff = P_hat_symm * self.adj
        cf_adj_actual = P * self.adj

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

        # edge_del equivalent
        cf_adj = P * self.adj

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
