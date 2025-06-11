import torch.nn.functional as func
import torch
import torch.nn as nn
from nidl.utils.dist import all_gather_batch_with_grad

class InfoNCE(nn.Module):
    """
    Normalized Temperature Cross-Entropy Loss for Contrastive Learning following [1].
    
    [1] A Simple Framework for Contrastive Learning of Visual Representations, Chen et al.,  ICML 2020
    """

    def __init__(self, temperature: float=0.1):
        """
        Parameters
        ----------
        temperature: float
            Scale logits by the inverse of the temperature.
        """


        super().__init__()
        self.temperature = temperature
        self.INF = 1e8

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
            The InfoNCE loss computed between `z1` and `z2`.
        """
        z1 = func.normalize(z1, p=2, dim=-1) # dim [N, D]
        z2 = func.normalize(z2, p=2, dim=-1) # dim [N, D]
        z1, z2 = all_gather_batch_with_grad([z1, z2])
        N = len(z1)
        sim_zii= (z1 @ z1.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z2 @ z2.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z1 @ z2.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z1.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z1.device)
        sim_Z = torch.cat([
            torch.cat([sim_zij, sim_zii], dim=1),
            torch.cat([sim_zjj, sim_zij.T], dim=1)], dim=0)
        log_sim_Z = func.log_softmax(sim_Z, dim=1)
        loss = - torch.diag(log_sim_Z).mean()
        return loss

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)
    
