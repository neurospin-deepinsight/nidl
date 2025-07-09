##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import torch

class BarlowTwins(nn.Module):
    """ Redundancy-reduction loss derived from Zbontar et al.,
    JMLR 2021. See https://proceedings.mlr.press/v139/zbontar21a for details.

    Parameters
    ----------
    lambd: float, default=5e-3
        trading off the importance of the redundancy reduction term.
    """

    def __init__(
            self,
            lambd: float = 5e-3):
        super().__init__()
        self.lambd = lambd

    def forward(
            self,
            z1: torch.Tensor,
            z2: torch.Tensor):
        """ Forward implementation.

        Parameters
        ----------
        z1: torch.Tensor of shape (batch_size, n_features)
            First embedded view.
        z2: torch.Tensor of shape (batch_size, n_features)
            Second embedded view.

        Returns
        ----------
        loss: torch.Tensor
            The BarlowTwins loss computed between `z1` and `z2`.
        """
        # normalize repr. along the batch dimension
        # beware: normalization is not robust to batch of size 1
        # if it happens, it will return a nan loss
        z1_norm = (z1 - z1.mean(0)) / z1.std(0) # NxD
        z2_norm = (z2 - z2.mean(0)) / z2.std(0) # NxD

        N = z1.size(0)
        D = z1.size(1)
        lbd = self.lambd / D

        # cross-correlation matrix
        c = torch.mm(z1_norm.T, z2_norm) / N # DxD
        # loss
        c_diff = (c - torch.eye(D,device=self.device)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambd
        c_diff[~torch.eye(D, dtype=bool)] *= lbd
        loss_invariance = c_diff[torch.eye(D, dtype=bool)].sum()
        loss_redundancy = c_diff[~torch.eye(D, dtype=bool)].sum()
        loss = loss_invariance + loss_redundancy

        return loss

    def __repr__(self):
        return f"{type(self).__name__}(lambd={self.lambd})"
    
