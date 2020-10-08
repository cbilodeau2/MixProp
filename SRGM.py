import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return -grad_output


class SRGM(nn.Module):

    def __init__(self, featurizer, loss_func, args):
        super(SRGM, self).__init__()
        self.featurizer = featurizer
        self.W_scaf = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.hidden_size),
        )
        self.perturb_forward = self.scaf_forward 

        self.hidden_size = args.hidden_size
        first_linear_dim = args.hidden_size
        if args.use_input_features:
            first_linear_dim += args.features_size

        self.classifier = nn.Sequential(
                nn.Dropout(args.dropout),
                nn.Linear(first_linear_dim, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.output_size),
        )
        self.copy_f_k = nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(args.dropout),
                    nn.Linear(first_linear_dim, args.hidden_size),
                    nn.ReLU(),
                    nn.Linear(args.hidden_size, args.output_size),
                ).requires_grad_(False) for _ in range(args.num_domains)
        ])
        self.f_k = nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(args.dropout),
                    nn.Linear(first_linear_dim, args.hidden_size),
                    nn.ReLU(),
                    nn.Linear(args.hidden_size, args.output_size),
                ) for _ in range(args.num_domains)
        ])
        self.g_k = nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(args.dropout),
                    nn.Linear(first_linear_dim, args.hidden_size),
                    nn.ReLU(),
                    nn.Linear(args.hidden_size, args.output_size),
                ) for _ in range(args.num_domains)
        ])
        self.h_k = nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(args.dropout),
                    nn.Linear(first_linear_dim, args.hidden_size),
                    nn.ReLU(),
                    nn.Linear(args.hidden_size, args.output_size),
                ) for _ in range(args.num_domains)
        ])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.loss_func = loss_func
        self.rgm_e = args.rgm_e
        self.rgm_p = args.rgm_p
        self.env_lr = args.rgm_env_lr
        self.num_domains = args.num_domains

    def scaf_forward(self, phi_x, batch_scaf):
        scaf_x = self.featurizer.encoder.encoder[0](batch_scaf[0])
        phi_x = self.W_scaf(phi_x[:, :self.hidden_size])  # only take the MPN outputs, no features
        scaf_x = self.W_scaf(scaf_x)
        logits = (phi_x.unsqueeze(1) * scaf_x.unsqueeze(0)).sum(dim=-1)
        label = torch.arange(len(phi_x)).cuda()
        return F.cross_entropy(logits, label)

    def dom_forward(self, phi_x, batch_d):
        logits = self.W_dom(phi_x)
        return self.dom_loss_func(logits, batch_d)

    def loss_forward(self, preds, batch_y, mask):
        pred_loss = self.loss_func(preds, batch_y) * mask
        return pred_loss.sum() / mask.sum()

    def perturb(self, phi_x, batch_d):
        scaf_loss = self.perturb_forward(phi_x, batch_d)
        scaf_grad = torch.autograd.grad([scaf_loss * len(batch_d)], [phi_x], create_graph=True)[0]
        new_x = phi_x + self.env_lr * scaf_grad.detach()  # increase scaffold classification loss
        return scaf_loss, new_x

    def forward(self, batches):
        for k in range(self.num_domains):
            self.copy_f_k[k].load_state_dict(self.f_k[k].state_dict())

        erm_loss = 0
        all_phis = []
        for batch_x, batch_f, batch_y, batch_d, mask in batches:
            phi_x = self.featurizer(batch_x, batch_f)
            all_phis.append(phi_x)
            preds = self.classifier(phi_x) 
            erm_loss = erm_loss + self.loss_forward(preds, batch_y, mask)

        regret = 0
        scaf_loss = 0
        for k in range(self.num_domains):
            batch_x, batch_f, batch_y, batch_d, mask = batches[k]
            phi_x = all_phis[k]
            preds = self.copy_f_k[k](phi_x) 
            oracle_preds = self.g_k[k](GradientReversal.apply(phi_x))
            regret = regret + self.loss_forward(preds, batch_y, mask) + self.loss_forward(oracle_preds, batch_y, mask)

            scaf_loss_k, new_x = self.perturb(phi_x, batch_d)
            scaf_loss = scaf_loss + scaf_loss_k

            preds = self.copy_f_k[k](new_x)
            oracle_preds = self.h_k[k](GradientReversal.apply(new_x))
            regret = regret + self.loss_forward(preds, batch_y, mask) + self.loss_forward(oracle_preds, batch_y, mask)
       
        holdout_loss = 0
        for k in range(self.num_domains):
            batch_x, batch_f, batch_y, batch_d, mask = batches[1 - k]  # hardcode: 2 domains
            phi_x = all_phis[1 - k].detach()  # phi does not help f_{-e}
            preds = self.f_k[k](phi_x)
            holdout_loss = holdout_loss + self.loss_forward(preds, batch_y, mask)

        loss = erm_loss + holdout_loss + self.rgm_e * regret + self.rgm_p * scaf_loss
        return loss / self.num_domains

