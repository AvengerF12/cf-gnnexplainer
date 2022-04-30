# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from utils.utils import get_degree_matrix
from .gcn_perturb_orig import GCNSyntheticPerturbOrig
from .gcn_perturb_delta import GCNSyntheticPerturbDelta
from .gcn_perturb_delta_CEM import GCNSyntheticPerturbCEM


class CFExplainer:
    """
    CF Explainer class, returns counterfactual subgraph
    """
    def __init__(self, model, cf_optimizer, lr, n_momentum, sub_adj, num_nodes, sub_feat,
                 n_hid, dropout, sub_label, num_classes, alpha, beta, gamma, task,
                 cem_mode=None, edge_del=False, edge_add=False, bernoulli=False, delta=False,
                 rand_init=True, history=False, div_hind=5, device=None, verbosity=0):

        super(CFExplainer, self).__init__()
        self.model = model
        self.cf_optimizer = cf_optimizer
        self.lr = lr
        self.n_momentum = n_momentum
        self.sub_adj = sub_adj
        self.num_nodes = num_nodes
        self.sub_feat = sub_feat
        self.n_hid = n_hid
        self.dropout = dropout
        self.sub_label = sub_label
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.task = task
        self.num_classes = num_classes
        self.cem_mode = cem_mode
        self.edge_del = edge_del
        self.edge_add = edge_add
        self.bernoulli = bernoulli
        self.delta = delta
        self.rand_init = rand_init
        self.history = history
        self.div_hind = div_hind
        self.device = device
        self.verbosity = verbosity

        self.model.eval()

        if self.cem_mode is None and not edge_del and not edge_add:
            raise RuntimeError("CFExplainer: need to specify allowed add/del op")

        if self.gamma > 0 and not self.history:
            raise RuntimeError("CFExplainer: gamma requires the history to be enabled")

        # Instantiate CF model class, load weights from original model
        if self.cem_mode == "PN" or self.cem_mode == "PP":
            self.cf_model = GCNSyntheticPerturbCEM(self.model, self.num_classes, self.sub_adj,
                                                   num_nodes, self.alpha, self.beta, self.gamma,
                                                   mode=self.cem_mode,
                                                   rand_init=self.rand_init,
                                                   device=self.device, task=self.task)

        elif self.cem_mode is None:

            if self.delta:
                self.cf_model = GCNSyntheticPerturbDelta(self.model, self.num_classes,
                                                         self.sub_adj, num_nodes, self.alpha,
                                                         self.beta, self.gamma,
                                                         edge_del=self.edge_del,
                                                         edge_add=self.edge_add,
                                                         bernoulli=self.bernoulli,
                                                         rand_init=self.rand_init,
                                                         device=self.device, task=self.task)
            else:
                self.cf_model = GCNSyntheticPerturbOrig(self.model, self.num_classes,
                                                        self.sub_adj, num_nodes, self.alpha,
                                                        self.beta, self.gamma,
                                                        edge_del=self.edge_del,
                                                        edge_add=self.edge_add,
                                                        bernoulli=self.bernoulli,
                                                        rand_init=self.rand_init,
                                                        device=self.device, task=self.task)
        else:
            raise RuntimeError("cf_explainer: the specified mode for CEM is invalid")

        if self.verbosity > 1:
            for name, param in self.model.named_parameters():
                print("orig model requires_grad: ", name, param.requires_grad)
            for name, param in self.cf_model.named_parameters():
                print("cf model required_grad: ", name, param.requires_grad)

        # Init optimizer used to generate explanation
        if cf_optimizer == "SGD" and n_momentum == 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr)
        elif cf_optimizer == "SGD" and n_momentum != 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr, nesterov=True,
                                          momentum=n_momentum)
        elif cf_optimizer == "Adadelta":
            self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=lr)


    def debug_check_expl(self, expl_example):

        # Check loss_graph_dist, handling edge case of PP which is not a CF
        if expl_example != [] and expl_example[-1] < 1 and self.cem_mode != "PP":
            error_str = "cf_explainer: loss_graph_dist cannot be smaller than 1. Check symmetry"
            raise RuntimeError(error_str)

        # Check cf_adj
        if expl_example != [] and 1 in torch.diagonal(expl_example[0], dim1=-2, dim2=-1):
            raise RuntimeError("cf_explainer: cf_adj contains a self-connection. Invalid result.")

        if expl_example != [] and torch.any(torch.greater(expl_example[0], 1)):
            raise RuntimeError("cf_explainer: cf_adj contains values > 1. Invalid result.")

        if expl_example != [] and torch.any(torch.less(expl_example[0], 0)):
            raise RuntimeError("cf_explainer: cf_adj contains values < 0. Invalid result.")


    def explain(self, task, num_epochs, y_pred_orig, node_idx=None, new_idx=None, debug=True):

        if task == "node-class" and (node_idx is None or new_idx is None):
            raise RuntimeError("cf_explainer/explain: invalid task")

        expl_list = []
        diff_adj_list = []
        best_loss = np.inf
        num_expl = 0

        for epoch in range(num_epochs):

            if task == "node-class":
                new_expl, cf_adj_diff, loss_graph_dist = self.train_expl(task, epoch, y_pred_orig,
                                                                         diff_adj_list, node_idx,
                                                                         new_idx)
            elif task == "graph-class":
                new_expl, cf_adj_diff, loss_graph_dist = self.train_expl(task, epoch, y_pred_orig,
                                                                         diff_adj_list)

            if self.verbosity > 1:
                print(loss_graph_dist, "(Current graph distance loss)")
                print(best_loss, "(Best distance loss)")

            if new_expl == []:
                continue

            # For PP, save every valid explanation generated
            cond_PP = self.cem_mode == "PP"
            # Else, save only explanations that are as good or better as the best ones
            cond_CF = loss_graph_dist <= best_loss

            if cond_PP or cond_CF:
                if self.history:
                    expl_list.append(new_expl)
                else:
                    expl_list = [new_expl]

                # Note: history cannot be false if gamma is greater than 0, otherwise error
                if self.gamma > 0:
                    diff_adj_list.append(cf_adj_diff)
                    diff_adj_list = diff_adj_list[-self.div_hind:]

                best_loss = loss_graph_dist
                num_expl += 1

            if debug:
                self.debug_check_expl(new_expl)

        expl_res = [node_idx, new_idx, expl_list, self.sub_adj.cpu(), self.sub_feat.cpu(),
                    self.sub_label.cpu(), y_pred_orig, self.num_nodes]

        return expl_res, num_expl


    def train_expl(self, task, epoch, y_pred_orig, prev_adj_list, node_idx=None, new_idx=None):
        self.cf_optimizer.zero_grad()

        output, output_actual = self.cf_model.forward(self.sub_feat)

        if task == "node-class":
            # Need to use new_idx from now on since sub_adj is reindexed
            output = output[new_idx]
            output_actual = output_actual[new_idx]

        y_pred_new = torch.argmax(output)
        y_pred_new_actual = torch.argmax(output_actual)

        if self.cem_mode == "PN":

            loss_total, loss_graph_dist, cf_adj_diff, cf_adj_actual = \
                self.cf_model.loss_PN(output, y_pred_orig, y_pred_new_actual, prev_adj_list)

        elif self.cem_mode == "PP":

            loss_total, loss_graph_dist, cf_adj_diff, cf_adj_actual = \
                self.cf_model.loss_PP(output, y_pred_orig, y_pred_new_actual, prev_adj_list)

        elif self.cem_mode is None:

            if self.bernoulli:
                loss_total, loss_graph_dist, cf_adj_diff, cf_adj_actual = \
                    self.cf_model.loss_bernoulli(output, y_pred_orig, y_pred_new_actual,
                                                 prev_adj_list)
            else:
                # loss_pred indicator function should be based on y_pred_new_actual NOT y_pred_new
                loss_total, loss_graph_dist, cf_adj_diff, cf_adj_actual = \
                    self.cf_model.loss_std(output, y_pred_orig, y_pred_new_actual, prev_adj_list)
        else:
            raise RuntimeError("cf_explainer/train: the specified mode for CEM is invalid")

        loss_total.backward()
        clip_grad_norm_(self.cf_model.parameters(), 2.0)
        self.cf_optimizer.step()

        if self.verbosity > 1:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss: {:.4f}'.format(loss_total.item()),
                  'graph loss: {:.4f}'.format(loss_graph_dist.item()),
                  'alpha (pred coeff): {},'.format(self.alpha),
                  'beta (dist coeff): {},'.format(self.beta),
                  'gamma (diverse expl coeff): {},'.format(self.gamma))
            print('Output: {}\n'.format(output.data),
                  'Output nondiff: {}\n'.format(output_actual.data),
                  'orig pred: {}, '.format(y_pred_orig),
                  'new pred: {}, '.format(y_pred_new),
                  'new pred nondiff: {}'.format(y_pred_new_actual))
            print(" ")

        # Note: when updating output format, also update checks
        expl_inst = []
        cond_PP = self.cem_mode == "PP" and y_pred_new_actual == y_pred_orig
        # Needed to avoid including PP with different predictions
        cond_cf = self.cem_mode != "PP" and y_pred_new_actual != y_pred_orig

        if cond_PP or cond_cf:
            expl_inst = [cf_adj_actual.detach().squeeze().cpu(), y_pred_new_actual.detach().cpu(),
                         loss_graph_dist.detach().item()]

        return(expl_inst, cf_adj_diff.detach(), loss_graph_dist.detach().item())
