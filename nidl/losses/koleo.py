##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import torch
import torch.nn.functional as func
from torch import Tensor
from torch.nn import Module


class KoLeoLoss(Module):
    """KoLeo regularizer [1]_, as in DINOv2 [2]_.

    ``-mean(log(d_i))``, with ``d_i`` the distance from the L2-normalized
    feature ``i`` to its nearest neighbor in the batch.

    Parameters
    ----------
    eps: float, default=1e-8
        Numerical stability constant.

    References
    ----------
    .. [1] Sablayrolles, A., et al., "Spreading vectors for similarity
           search." ICLR, 2019. https://arxiv.org/abs/1806.03198
    .. [2] Oquab, M., et al., "DINOv2: Learning Robust Visual Features
           without Supervision." TMLR, 2024. https://arxiv.org/abs/2304.07193
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Compute the loss.

        Parameters
        ----------
        x: torch.Tensor of shape (batch_size, n_features)
            Features.

        Returns
        -------
        loss: torch.Tensor
            The loss.
        """
        x = func.normalize(x.float(), dim=-1, eps=self.eps)
        with torch.no_grad():
            sim = x @ x.T
            sim.fill_diagonal_(-2)  # exclude self
            neighbor = sim.argmax(dim=-1)
        dist = func.pairwise_distance(x, x[neighbor], eps=self.eps)
        return -torch.log(dist + self.eps).mean()
