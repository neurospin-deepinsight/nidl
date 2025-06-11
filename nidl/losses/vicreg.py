import torch.nn.functional as F
import torch
import torch.nn as nn
from nidl.utils.dist import all_gather_batch_with_grad

class VICRegLoss(nn.Module):
    """
    Variance-Invariance-Covariance Regularization (VICReg) loss following [1].

    The implementation is based on the official code: https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
    
    [1] VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning, ICLR 2022
    """

    def __init__(self, 
                 sim_coeff: float=25.0, 
                 std_coeff: float=25.0, 
                 cov_coeff: float=1.0):
        """
        Parameters
        ----------
        sim_coeff: float, default=25.0
            Invariance regularization loss coefficient.
        
        std_coeff: float, default=25.0
            Variance regularization loss coefficient.

        cov_coeff: float, default=1.0
            Covariance regularization loss coefficient.
        """
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        """
        Parameters
        ----------
        z1: torch.Tensor of shape (batch_size, n_features)
            First embedded view.

        z2: torch.Tensor of shape (batch_size, n_features)
            Second embedded view.

        Returns
        ----------
        loss: torch.Tensor
            The VICReg loss computed between `z1` and `z2`.
        """
        z1, z2 = all_gather_batch_with_grad([z1, z2])

        bs, num_features = z1.shape

        inv_loss = F.mse_loss(z1, z2)

        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        
        std_z1 = torch.sqrt(z1.var(dim=0) + 0.0001)
        std_z2 = torch.sqrt(z2.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_z1)) / 2 + torch.mean(F.relu(1 - std_z2)) / 2

        cov_z1 = (z1.T @ z1) / (bs - 1)
        cov_z2 = (z2.T @ z2) / (bs - 1)
        cov_loss = self.off_diagonal(cov_z1).pow_(2).sum().div(
            num_features
        ) + self.off_diagonal(cov_z2).pow_(2).sum().div(num_features)

        loss = (
            self.sim_coeff * inv_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def __str__(self):
        return "{}(sim={}, std={}, cov={})".format(
            type(self).__name__, self.sim_coeff, self.std_coeff, self.cov_coeff)

