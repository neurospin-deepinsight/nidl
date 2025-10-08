##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import torch
import torch.nn as nn
import torch.nn.functional as func


class InfoNCE(nn.Module):
    """Normalized temperature cross-entropy loss derived from Chen et al.,
    ICML 2020. See https://doi.org/10.48550/arXiv.2002.05709 for details.

    Parameters
    ----------
    temperature: float, default=0.1
        scale logits by the inverse of the temperature.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        """Forward implementation.

        Parameters
        ----------
        z1: torch.Tensor of shape (batch_size, n_features)
            First embedded view.
        z2: torch.Tensor of shape (batch_size, n_features)
            Second embedded view.

        Returns
        -------
        loss: torch.Tensor
            The InfoNCE loss computed between `z1` and `z2`.
        """
        # Concatenate features
        feats = torch.cat([z1, z2], dim=0)
        # Calculate cosine similarity
        cos_sim = func.cosine_similarity(
            feats[:, None, :], feats[None, :, :], dim=-1
        )
        # Mask out cosine similarity to itself
        self_mask = torch.eye(
            cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device
        )
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        return nll.mean()

    def __repr__(self):
        return f"{type(self).__name__}(temperature={self.temperature})"
