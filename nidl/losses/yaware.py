import torch.nn as nn
import torch
import numpy as np
from typing import Optional
# Local imports
from nidl.utils.dist import all_gather_batch_with_grad
from nidl.models.ssl.utils.kernels import KernelMetric
from nidl.utils.similarity import PairwiseCosineSimilarity

class yAwareInfoNCE(nn.Module):
    """
    Implementation of y-Aware InfoNCE loss [1]. 

    [1] Contrastive Learning with Continuous Proxy Meta-Data for 3D MRI Classification, MICCAI 2021
    """

    def __init__(self, 
                 kernel: KernelMetric,
                 temperature: float = 0.1):
        """
        Parameters
        ----------
        kernel: neuroclav.utils.kernels.KernelMetric
            Kernel to compute the similarity matrix between auxiliary variables.
            See PhD thesis, Dufumier 2022 page 94-95.
        
        temperature: float, default=0.1
            Temperature used to scale the dot-product between embedded input vectors
        """
        super().__init__()
        if kernel is not None and not isinstance(kernel, KernelMetric):
            raise ValueError("`kernel` must be KernelMetric object (got %s)"%type(kernel))
        self.kernel = kernel
        self.sim_metric = PairwiseCosineSimilarity()
        self.temperature = temperature
        self.INF = 1e8


    def forward(self, 
                z1: torch.Tensor, 
                z2: torch.Tensor, 
                labels: Optional[torch.Tensor]=None):
        """
        Parameters
        ----------
        z1: torch.Tensor of shape (batch_size, n_features)
            First embedded view.

        z2: torch.Tensor of shape (batch_size, n_features)
            Second embedded view.
        
        labels: Optional[torch.Tensor] of shape (batch_size, n_labels)
            Auxiliary variables associated to the input data.
            If None, the standard InfoNCE loss is returned.

        Returns
        ----------
        loss: torch.Tensor
            The y-Aware InfoNCE loss computed between `z1` and `z2`.
        """

        N = len(z1)
        assert len(z1) == len(z2), "Two tensors z1, z2 must have " \
                                   "same shape, got {} != {}".format(z1.shape, z2.shape)
        if labels is not None:
            assert N == len(labels), "Labels length %i != vectors length %i"%(len(labels), N)

        z1, z2 = all_gather_batch_with_grad([z1, z2])

        # Computes similarity matrices, shape (N, N)
        sim_z11= self.sim_metric(z1, z1) / self.temperature
        sim_z22 = self.sim_metric(z2, z2) / self.temperature
        sim_z12 = self.sim_metric(z1, z2) / self.temperature
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_z11 = sim_z11 - self.INF * torch.eye(N, device=z1.device)
        sim_z22 = sim_z22 - self.INF * torch.eye(N, device=z2.device)

        # Stack similarity matrices, shape (2N, 2N)
        sim_Z = torch.cat([torch.cat([sim_z11, sim_z12], dim=1),
                           torch.cat([sim_z12.T, sim_z22], dim=1)], dim=0)

        if self.kernel is None or labels is None:
            correct_pairs = torch.arange(N, device=z1.device).long()
            loss_1 = nn.functional.cross_entropy(torch.cat([sim_z12, sim_z11], dim=1), correct_pairs)
            loss_2 = nn.functional.cross_entropy(torch.cat([sim_z12.T, sim_z22], dim=1), correct_pairs)
            loss = (loss_1 + loss_2) / 2.
        else:
            all_labels = labels.view(N, -1).repeat(2, 1).detach().cpu().numpy() # [2N, *]
            weights = self.kernel.pairwise(all_labels).astype(float) # [2N, 2N]
            weights = weights * (1 - np.eye(2*N)) # Put zeros on the diagonal
            weights /= weights.sum(axis=1)
            log_sim_Z = nn.functional.log_softmax(sim_Z, dim=1)
            loss = -1./N * (torch.from_numpy(weights).to(z1.device) * log_sim_Z).sum()
        return loss

    def __str__(self):
        return "{}(temp={}, kernel={} ({})".format(
            type(self).__name__, self.temperature, self.kernel)
