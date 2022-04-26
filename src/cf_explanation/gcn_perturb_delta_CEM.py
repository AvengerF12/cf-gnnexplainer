import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils.utils import get_degree_matrix, BernoulliMLSample, create_symm_matrix_tril


# This class implements the CEM framework adapted to the GCN model
# It uses the delta formulation of the problem together with the bernoulli appraoch to P
# The actions allowed are implicit to the choice of explanation:
#   - Pertinent positive (PP): only edge deletion allowed
#   - Pertinent negative (PN): only edge addition allowed
# Note: the class is almost identical to *_delta, the only change is the loss_PP
# It has been duplicated in order to improve readability
class GCNSyntheticPerturbCEM(nn.Module):
    """
    3-layer GCN used in GNN Explainer synthetic tasks
    """
    def __init__(self, model, nclass, adj, num_nodes, beta, task, mode="PN", rand_init=True,
                 device=None):

        super(GCNSyntheticPerturbCEM, self).__init__()
        self.model = model
        # The adj mat is stored since each instance of the explainer deals with a single node
        self.adj = adj
        self.nclass = nclass
        self.beta = beta
        self.BML = BernoulliMLSample.apply

        # Used to find the appropriate part of P to perturbate (w/o padding) and compute flattened
        # layer for graph classification (differs from num_nodes_adj only when task = graph-class)
        self.num_nodes_actual = num_nodes

        self.mode = mode
        self.device = device

        # Number of nodes in the adj, in case of graph-class includes padding
        self.num_nodes_adj = self.adj.shape[1]

        # The optimizer will affect only the elements below the diag of this matrix
        # Note: no diagonal, it is assumed to be always 0/no self-connections allowed
        self.P_tril = Parameter(torch.FloatTensor(torch.zeros(self.num_nodes_actual,
                                                              self.num_nodes_actual)))
        if rand_init:
            torch.nn.init.uniform_(self.P_tril, a=0, b=0.4)

        # Avoid creating an eye matrix for each normalize_adj op, re-use the same one
        self.norm_eye = torch.eye(self.num_nodes_adj, device=device)


    def forward(self, x):

        output = None

        if self.mode == "PN":
            output = self.__forward_PN(x)
        elif self.mode == "PP":
            output = self.__forward_PP(x)
        else:
            raise RuntimeError("forward(CEM): invalid mode specified")

        return output, output


    def __forward_PN(self, x):

        P_hat_symm = create_symm_matrix_tril(self.P_tril, self.num_nodes_adj)
        P = self.BML(P_hat_symm)  # Threshold P_hat

        # edge_add equivalent
        delta = (1 - self.adj) * P
        A_tilde = self.adj + delta

        # Note: identity matrix is added in normalize_adj() inside model
        output = self.model(x, A_tilde.expand(1, -1, -1)).squeeze()

        return output


    def __forward_PP(self, x):

        P_hat_symm = create_symm_matrix_tril(self.P_tril, self.num_nodes_adj)
        P = self.BML(P_hat_symm)  # Threshold P_hat

        # edge_del equivalent
        delta = P * self.adj
        A_tilde = self.adj - delta

        # Note: identity matrix is added in normalize_adj() inside model
        output = self.model(x, A_tilde.expand(1, -1, -1)).squeeze()

        return output


    def loss_PN(self, output, y_pred_orig, y_pred_new_actual):
        P_hat_symm = create_symm_matrix_tril(self.P_tril, self.num_nodes_adj)
        P = self.BML(P_hat_symm)  # Threshold P_hat

        pred_same = (y_pred_new_actual == y_pred_orig).float()

        # edge_add equivalent
        delta = (1 - self.adj) * P
        cf_adj = self.adj + delta

        # Want negative in front to maximize loss instead of minimizing it to find CFs
        loss_pred = - F.nll_loss(output, y_pred_orig)
        # Number of edges changed (symmetrical)
        loss_graph_dist = torch.sum(torch.abs(delta)) / 2

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist

        # In this case it could be handy to return the delta directly instead of cf_adj,
        # in any case it is easy to find the delta given cf_adj and sub_adj
        return loss_total, loss_pred, loss_graph_dist, cf_adj


    def loss_PP(self, output, y_pred_orig, y_pred_new_actual):
        P_hat_symm = create_symm_matrix_tril(self.P_tril, self.num_nodes_adj)
        P = self.BML(P_hat_symm)  # Threshold P_hat

        # Note: flipped the boolean since we want the same prediction
        pred_diff = (y_pred_new_actual != y_pred_orig).float()

        # edge_del equivalent
        delta = self.adj * P
        cf_adj = self.adj - delta

        # Note: the negative sign is gone since we want to keep the same prediction
        loss_pred = F.nll_loss(output, y_pred_orig)
        # Number of edges in neighbourhood (symmetrical)
        # Note: here we are interested in finding the most sparse cf_adj with the same pred
        loss_graph_dist = torch.sum(torch.abs(cf_adj)) / 2
        # Note: in order to generate the best PP we need to minimize the number of entries in the
        # cf_adj, however in order to get a better understanding the number of edges deleted is 
        # more useful to the end user. Therefore this number is the one saved inside each cf_example
        # generated
        loss_graph_dist_actual = torch.sum(torch.abs(delta)) / 2

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = pred_diff * loss_pred + self.beta * loss_graph_dist

        return loss_total, loss_pred, loss_graph_dist_actual, cf_adj
