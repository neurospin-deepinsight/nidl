##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from collections.abc import Callable
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch import Tensor


class DCL(nn.Module):
    """Implementation of the Decoupled Contrastive Loss [1]_

    This loss function implements the decoupled contrastive loss as described
    in [1]_. It builds upon the classic InfoNCE loss but removes the positive-
    negative coupling that biases training in small batch sizes.

    Given a mini-batch of size :math:`N`, we obtain two embeddings
    :math:`z_{i}^(1)` and :math:`z_{i}^(2)` representing two different
    augmented views of the same sample. The **DCL** loss is defined as:

    .. math::
        \\mathcal{L}_i^{(k)}
        = - \\big(\\operatorname{sim}(z_i^{(1)}, z_i^{(2)})/\\tau\\big)
        + \\log
        \\sum\\limits_{l \\in \\{1,2\\}, j \\in \\![1,N\\!]}
        \\mathbf{1}_{[j \\ne i]},
        \\exp\\!\\big(\\operatorname{sim}(z_i^{(k)}, z_j^{(l)})/\\tau\\big)

    where :math:`\\operatorname{sim}(z_i^(k), z_j^(l))` denotes the cosine
    similarity between the normalized embeddings :math:`z_i^(k)` and
    :math:`z_j^(l)`, and :math:`\\tau > 0` is a temperature parameter
    controlling the concentration of the distribution.
    :math:`\\mathbf{1}_{[j \\ne i]}` ensures decoupling.

    Additionnaly, a weighting function :math:`w` can be added to modulate the
    contribution of the positive pairs' similarity to the loss. The intuition
    is that when the embedding of the positive sample :math:`z_i^{(2)}` is
    close to the anchor :math:`z_i^{(1)}`, there is less learning signal than
    when the two embeddings are less similar. The weighted loss is:

    .. math::
        \\mathcal{L}_i^{(k)}
        = - w(z_i^{(1)}, z_i^{(2)})
        \\big(\\operatorname{sim}(z_i^{(1)}, z_i^{(2)})/\\tau\\big)
        + \\log
        \\sum\\limits_{l \\in \\{1,2\\}, j \\in \\![1,N\\!]}
        \\mathbf{1}_{[j \\ne i]},
        \\exp\\!\\big(\\operatorname{sim}(z_i^{(k)}, z_j^{(l)})/\\tau\\big)

    See the class `DCLW` for an implementation with a negative von Mises-Fisher
    weighting function such as proposed in [1]_.

    Parameters
    ----------
    temperature: float, default=0.1
        Scale logits by the inverse of the temperature.
    pos_weight_fn: Optional[callable], default=None
        Weighting function of the positive pairs (:math:`w` in [1]_).
        If None, a DCL loss without weighting is returned.


    References
    ----------
    .. [1] Yeh, Chun-Hsiao, et al. "Decoupled contrastive learning."
           European conference on computer vision.
           Cham: Springer Nature Switzerland,
           https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860653.pdf

    """

    def __init__(self,
            temperature: float = 0.1,
            pos_weight_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
          ):
        super().__init__()
        # Check parameters
        if temperature < 0:
            raise ValueError("temperature parameter should be "
                             f"positive (got {temperature})")
        if not (isinstance(pos_weight_fn, Callable) or pos_weight_fn is None):
            raise ValueError("pos_weight_fn should be None or a callable "
                             "function that takes 2 tensors as input and "
                             "outputs a tensor.")

        self.temperature = temperature
        self.pos_weight_fn = pos_weight_fn

    def forward(self, z1: Tensor, z2: Tensor):
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
            The DCL loss computed between `z1` and `z2`.
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
        # Scale by temperature parameter
        cos_sim = cos_sim / self.temperature
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
    
        # Extract similarity from positive pairs and apply weights
        pos_sim = cos_sim[pos_mask]
        if self.pos_weight_fn is not None:
            pos_sim = self.pos_weight_fn(z1, z2).repeat(2) * pos_sim
        # Extract similarity from negative pairs only (here is the decoupling)
        neg_sim = cos_sim.masked_fill(pos_mask, -9e15)
        # DCL loss
        loss = (-pos_sim + torch.logsumexp(neg_sim, dim=-1)).mean()

        return loss

    def __repr__(self):
        return (f"{type(self).__name__}(temperature={self.temperature}, "
            f"pos_weight_fn={self.pos_weight_fn})")

class DCLW(DCL):
    """Decoupled Contrastive Loss (DCL) with von Mises-Fisher (vMF) weighting.

    It implements the DCL with vMF weighting as described in [1]_.
    See the documentation for `DCL` for more details.

    The vMF weighting function is defined as:

    .. math::
        w(z_i^{(1)}, z_i^{(2)})
        = 2 -
        \\frac{
        \\exp\\!\\big(\\operatorname{sim}(z_i^{(1)}, z_i^{(2)})/\\sigma\\big)
        }{
        \\frac{1}{N}\\sum\\limits_{j=1}{N}
        \\exp\\!\\big(\\operatorname{sim}(z_i^{(1)}, z_i^{(2)})/\\sigma\\big)
        }

    where :math:`N` is the batch size,
    :math:`\\operatorname{sim}(z_i^(1), z_i^(2))`
    denotes the cosine similarity between the normalized embeddings
    :math:`z_i^(1)` and :math:`z_i^(2)`, and :math:`\\sigma > 0` is a
    temperature parameter controlling the concentration of the distribution.

    Parameters
    ----------
    sigma: float, default=0.5
        Temperature parameter of the von Mises-Fisher weighting function.
    temperature: float, default=0.1
        Scale logits by the inverse of the temperature.

    References
    ----------
    .. [1] Yeh, Chun-Hsiao, et al. "Decoupled contrastive learning."
           European conference on computer vision.
           Cham: Springer Nature Switzerland, 2022.
           https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860653.pdf

    """

    def __init__(self, sigma=0.5, temperature=0.1):
        # Define the negative von Mises-Fisher weighting function
        def neg_von_mises_fisher(z1,z2):
            cos_sim = func.cosine_similarity(z1, z2, dim=1) / sigma
            return 2 - z1.shape[0] * func.softmax(cos_sim, dim=0)
        super().__init__(pos_weight_fn=neg_von_mises_fisher,
                         temperature=temperature)