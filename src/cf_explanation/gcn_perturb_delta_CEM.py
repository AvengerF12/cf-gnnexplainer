import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils.utils import get_degree_matrix, normalize_adj, BernoulliMLSample, create_symm_matrix_tril
from gcn import GraphConvolution, GCNSynthetic


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
    def __init__(self, nfeat, nhid, nout, nclass, adj, dropout, beta, mode="PN", device=None):

        super(GCNSyntheticPerturbCEM, self).__init__()
        # The adj mat is stored since each instance of the explainer deals with a single node
        self.adj = adj
        self.nclass = nclass
        self.beta = beta
        self.num_nodes = self.adj.shape[0]
        self.BML = BernoulliMLSample.apply
        self.mode = mode
        self.device = device

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

        output = None

        if self.mode == "PN":
            output = self.__forward_PN(x)
        elif self.mode == "PP":
            output = self.__forward_PP(x)
        else:
            raise RuntimeError("forward(CEM): invalid mode specified")

        return output, output


    def __forward_PN(self, x):

        P_hat_symm = create_symm_matrix_tril(self.P_tril)
        P = self.BML(P_hat_symm)  # Threshold P_hat

        # edge_add equivalent
        delta = (1 - self.adj.int()) * P
        A_tilde = self.adj + delta

        # Note: identity matrix is added in normalize_adj()
        norm_adj = normalize_adj(A_tilde, self.norm_eye, self.device)

        output = self.__apply_model(x, norm_adj)
        act_output = F.log_softmax(output, dim=1)

        return act_output


    def __forward_PP(self, x):

        P_hat_symm = create_symm_matrix_tril(self.P_tril)
        P = self.BML(P_hat_symm)  # Threshold P_hat

        # edge_del equivalent
        delta = P * self.adj
        A_tilde = self.adj - delta

        # Note: identity matrix is added in normalize_adj()
        norm_adj = normalize_adj(A_tilde, self.norm_eye, self.device)

        output = self.__apply_model(x, norm_adj)
        act_output = F.log_softmax(output, dim=1)

        return act_output


    def loss_PN(self, output, y_pred_orig, y_pred_new_actual):
        P_hat_symm = create_symm_matrix_tril(self.P_tril)
        P = self.BML(P_hat_symm)  # Threshold P_hat

        pred_same = (y_pred_new_actual == y_pred_orig).float()

        # edge_add equivalent
        delta = (1 - self.adj.int()) * P
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
        P_hat_symm = create_symm_matrix_tril(self.P_tril)
        P = self.BML(P_hat_symm)  # Threshold P_hat

        # Note: flipped the boolean since we want the same prediction
        pred_diff = (y_pred_new_actual != y_pred_orig).float()

        # edge_del equivalent
        delta = self.adj.int() * P
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
