##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from typing import Optional

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Base class for all projection and prediction heads in self-supervised
    estimators.

    Parameters
    ----------
    blocks : list of tuple (int, int, Optional[nn.Module], Optional[nn.Module])
        List of tuples, each denoting one block of the projection head MLP.
        Each tuple reads `(in_features, out_features, batch_norm_layer,
        non_linearity_layer)`. Each block applies:

        1) a linear layer with `in_features` and `out_features` (with bias if
           `batch_norm_layer` is None)
        2) a batch normalization layer as defined by `batch_norm_layer`
            (optional)
        3) a non-linearity as defined by `non_linearity_layer` (optional)

    Attributes
    ----------
    layers : nn.Sequential
        List of :class:`~torch.nn.Module` to apply.

    Examples
    --------
    >>> # the following projection head has two blocks
    >>> # the first block uses batch norm an a ReLU non-linearity
    >>> # the second block is a simple linear layer
    >>> projection_head = ProjectionHead([
    >>>     (256, 256, nn.BatchNorm1d(256), nn.ReLU()),
    >>>     (256, 128, None, None)
    >>> ])
    """

    def __init__(
        self,
        blocks: list[
            tuple[int, int, Optional[nn.Module], Optional[nn.Module]]
        ],
    ):
        super().__init__()

        layers = []
        for input_dim, output_dim, batch_norm, non_linearity in blocks:
            use_bias = not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Computes one forward pass through the projection head."""
        return self.layers(x)


class SimCLRProjectionHead(ProjectionHead):
    """Projection head used for SimCLR.

    This module implements the projection head as described in SimCLR [1]_.
    The projection head is a multilayer perceptron (MLP) with one hidden layer
    and a ReLU non-linearity, defined as:

    .. math::
        \\mathbf{z} = g(\\mathbf{h}) = W_2 \\cdot \sigma(W_1\\cdot\\mathbf{h})

    where :math:`\\sigma` is the ReLU activation function.

    References
    ----------
    .. [1] Chen, T., et al. "A Simple Framework for Contrastive Learning of
           Visual Representations." ICML, 2020. https://arxiv.org/abs/2002.05709
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        output_dim: int = 128,
    ):
        super().__init__(
            [
                (input_dim, hidden_dim, None, nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


class YAwareProjectionHead(ProjectionHead):
    """Projection head used for yAware contrastive learning.

    This module implements the projection head :math:`z_{\\theta_2}` as
    described in yAware [1]_, which is a simple multilayer perceptron (MLP)
    similar to that used in SimCLR [2]_. It maps feature representations into
    a space where contrastive loss can be applied.

    Typically, this MLP consists of one hidden layer followed by a
    non-linearity (ReLU) and a final linear projection.

    References
    ----------
    .. [1] Dufumier, B., et al., "Contrastive learning with continuous proxy
           meta-data for 3D MRI classification." MICCAI, 2021.
           https://arxiv.org/abs/2106.08808

    .. [2] Chen, T., et al. "A Simple Framework for Contrastive Learning of
           Visual Representations." ICML, 2020. https://arxiv.org/abs/2002.05709

    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        output_dim: int = 128,
    ):
        super().__init__(
            [
                (input_dim, hidden_dim, None, nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


class BarlowTwinsProjectionHead(ProjectionHead):
    """Projection head used for Barlow Twins [1]_.

    It implements the upscaling of layer sizes
    (hidden and output layers of size 8192),
    with 3-layer MLPs as in [1]_ or [2]_

    References
    ----------
    .. [1] Zbontar, J., et al., "Barlow Twins: Self-Supervised Learning
           via Redundancy Reduction." PMLR, 2021.
           https://proceedings.mlr.press/v139/zbontar21a
    .. [2] Siddiqui, S., et al., "Blockwise Self-Supervised Learning at Scale"
           TMLR, 2024.
           https://openreview.net/forum?id=M2m618iIPk

    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 8192,
        output_dim: int = 8192,
    ):
        super().__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (
                    hidden_dim,
                    hidden_dim,
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                ),
                (hidden_dim, output_dim, None, None),
            ]
        )


class DINOProjectionHead(ProjectionHead):
    """Projection head used in DINO [1]_.

    The projection head consists of a 3-layer multi-layer perceptron (MLP)
    with hidden dimension 2048 followed by l2 normalization and a weight
    normalized fully connected layer with K dimensions, which is similar to the
    design from SwAV [2]_.

    Parameters
    ----------
    input_dim: int, default=2048
        The input dimension of the head.
    hidden_dim: int, default=2048
        The hidden dimension.
    bottleneck_dim: int, default=256
        Dimension of the bottleneck in the last layer of the head.
    output_dim: int, default=4096
        The output dimension of the head.
    batch_norm: bool, default=True
        Whether to use batch norm or not. Should be set to False when using
        a vision transformer backbone.
    freeze_last_layer: int, default=-1
        Number of epochs during which we keep the output layer fixed.
        Typically doing so during the first epoch helps training. Try
        increasing this value if the loss does not decrease.
    norm_last_layer: bool, default=True
        Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the
        training unstable.

    References
    ----------
    .. [1] Caron, M., et al., "Emerging Properties in Self-Supervised Vision
           Transformers" ICCV, 2021. https://arxiv.org/abs/2104.14294
    .. [2] Caron, M., et al., "Unsupervised Learning of Visual Features by
           Contrasting Cluster Assignments", NeurIPS, 2020. https://arxiv.org/abs/2006.09882
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 4096,
        batch_norm: bool = True,
        freeze_last_layer: int = -1,
        norm_last_layer: bool = True,
    ):
        """Initializes the DINOProjectionHead with the specified dimensions."""
        super().__init__(
            [
                (
                    input_dim,
                    hidden_dim,
                    nn.BatchNorm1d(hidden_dim) if batch_norm else None,
                    nn.GELU(),
                ),
                (
                    hidden_dim,
                    hidden_dim,
                    nn.BatchNorm1d(hidden_dim) if batch_norm else None,
                    nn.GELU(),
                ),
                (hidden_dim, bottleneck_dim, None, None),
            ]
        )
        self.apply(self._init_weights)
        self.freeze_last_layer = freeze_last_layer
        self.last_layer = nn.Linear(bottleneck_dim, output_dim, bias=False)
        self.last_layer = nn.utils.weight_norm(self.last_layer)
        self.last_layer.weight_g.data.fill_(1)

        # Option to normalize last layer.
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def cancel_last_layer_gradients(self, current_epoch: int) -> None:
        """Cancel last layer gradients to stabilize the training."""
        if current_epoch >= self.freeze_last_layer:
            return
        for param in self.last_layer.parameters():
            param.grad = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes one forward pass through the head."""
        x = self.layers(x)
        # l2 normalization
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes layers with a truncated normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init._no_grad_trunc_normal_(
                module.weight,
                mean=0,
                std=0.02,
                a=-2,
                b=2,
            )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
