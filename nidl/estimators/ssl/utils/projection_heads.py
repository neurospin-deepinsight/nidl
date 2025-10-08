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
