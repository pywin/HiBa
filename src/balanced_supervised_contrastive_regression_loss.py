import torch
import torch.nn as nn
from math import log

class BalancedSupervisedContrastiveRegressionLoss(nn.Module):
    def __init__(self, temperature=0.07):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        """
        super(BalancedSupervisedContrastiveRegressionLoss, self).__init__()
        self.temperature = temperature

    def forward(self, projections, targets, weights):
        """
        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :param centers: torch.Tensor, shape [121, 3, 512]
        :param bank: torch.Tensor, shape [bank_size, feat_dim]
        :return: torch.Tensor, scalar
        """
        targets = targets.to(torch.int64)
        targets = targets-40
        targets_weight = torch.take(weights, targets)
        targets_weight_mtx = torch.mul(targets_weight.unsqueeze(1), targets_weight.T)

        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")

        dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        cardinality_per_samples = torch.sum(mask_anchor_out, dim=1)

        dist_ij = torch.abs(targets.reshape(-1, 1) - targets).to(device)
        dist_ij_ik = torch.repeat_interleave(dist_ij, dist_ij.size(1)).view(-1, dist_ij.size(0), dist_ij.size(1))
        dist_mask = (dist_ij.unsqueeze(1) >= dist_ij_ik).to(device)
        denominator = exp_dot_tempered * dist_mask * targets_weight_mtx
        log_prob = -torch.log(exp_dot_tempered * targets_weight_mtx / (torch.sum(denominator, dim=[2])))

        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_anchor_out, dim=1) / (cardinality_per_samples + 1e-5)
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss
