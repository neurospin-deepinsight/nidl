##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import torch
import torch.nn as nn


class BarlowTwins(nn.Module):
    """Implementation of the redundancy-reduction loss [1]_.

    Compute the Barlow Twins loss, which reduces redundancy
    between the components of the outputs.

    Given a mini-batch of size :math:`n`, two embeddings
    :math:`z^{(1)}_b` and :math:`z^{(2)}_b` representing
    two outputs of dimension :math:`D` of the same sample:

    .. math::`
        \mathcal{L}_{BT} \\triangleq
        \\underbrace{\sum_{i} \\left( 1 - C_{ii} \\right)^{2}
        }_{\\text{invariance term}}
        + \\frac{\lambda}{D} \,
        \\underbrace{\sum_{i} \sum_{j \\neq i} C_{ij}^{2}
        }_{\\text{redundancy reduction term}}

    where :math:`\\lambda` is a positive constant trading off
    the importance of the first and second terms of the loss,
    and where :math:`C` is the cross-correlation matrix computed
    between the outputs of the two identical networks
    along the batch dimension:

    .. math::
        C_{ij} \\triangleq
        \\frac{\sum_{b} z^{(1)}_{b,i} \, z^{(2)}_{b,j}}
        {\\sqrt{\sum_{b} \\left(z^{(1)}_{b,i}\\right)^{2}}
        \; \\sqrt{\sum_{b} \\left(z^{(2)}_{b,j}\\right)^{2}} }

    where :math:`b` indexes batch samples
    and :math:`i, j` index the vector dimension of the networks outputs.

    Parameters
    ----------
    lambd: float, default=5e-3
        trading off the importance of the redundancy reduction term.
        In the loss, it is divided by the :math:`D`,
        the dimension of the output


    References
    ----------
    .. [1] Zbontar, J., et al., "Barlow Twins: Self-Supervised Learning
           via Redundancy Reduction." PMLR, 2021.
           hhttps://proceedings.mlr.press/v139/zbontar21a

    """

    def __init__(self, lambd: float = 5e-3):
        super().__init__()
        self.lambd = lambd

    def forward(
            self,
            z1: torch.Tensor,
            z2: torch.Tensor
    ):
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
            The BarlowTwins loss computed between `z1` and `z2`.
        """
        # normalize repr. along the batch dimension
        # beware: normalization is not robust to batch of size 1
        # if it happens, it sets corresponding std deviation to 1
        N = z1.size(0)
        D = z1.size(1)
        if N == 1:
            z1_norm = (z1 - z1.mean(0))
            z2_norm = (z2 - z2.mean(0))
        else:
            z1_norm = (z1 - z1.mean(0)) / z1.std(0)  # NxD
            z2_norm = (z2 - z2.mean(0)) / z2.std(0)  # NxD


        lbd = self.lambd / D

        # cross-correlation matrix
        c = torch.mm(z1_norm.T, z2_norm) / (N - 1)  # DxD

        # loss
        c_diff = (c - torch.eye(D, device=z1.device)).pow(2)  # DxD

        # multiply off-diagonal elems of c_diff by lambd
        c_diff[~torch.eye(D, dtype=bool)] *= lbd
        loss_invariance = c_diff[torch.eye(D, dtype=bool)].sum()
        loss_redundancy = c_diff[~torch.eye(D, dtype=bool)].sum()
        loss = loss_invariance + loss_redundancy

        return loss

    def __str__(self):
        return f"{type(self).__name__}(lambd={self.lambd})"

