##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

from .....transforms import TypeTransformInput, VolumeTransform


class ZNormalization(VolumeTransform):
    """Normalize a 3d volume by removing the mean and scaling to unit variance.

    Applies the following normalization to each channel separately:

    .. math::

        x_i' = \\frac{x_i - \\mu(x)}{\\sigma(x)+\\epsilon}

    where :math:`x_i` is the original voxel intensity, :math:`\\mu(x)`
    is the data mean, :math:`\\sigma(x)` is the data std, and
    :math:`\\epsilon` is a small constant added for numerical stability.

    It can handle a :class:`numpy.ndarray` or :class:`torch.Tensor` as input
    and it returns a consistent output (same type and shape). Input shape must
    be :math:`(C, H, W, D)` or :math:`(H, W, D)` (spatial dimensions).


    Parameters
    ----------
    masking_fn: Callable or None, default=None
        If Callable, a masking function to be applied on the input data
        for each channel separately. It should return a boolean mask
        used to compute the data statistics (mean and std).
        If None, the whole volume is taken to compute the statistics.

    eps: float, default=1e-8
        Small float added to the standard deviation to avoid numerical
        errors.

    kwargs: dict
        Keyword arguments given to :class:`nidl.transforms.Transform`.

    Notes
    -----
    If the input volume has constant values, the output will have almost
    constant non-deterministic values.

    """

    def __init__(
        self,
        masking_fn: Optional[Callable] = None,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.masking_fn = masking_fn
        self.eps = eps

    def apply_transform(self, data: TypeTransformInput) -> TypeTransformInput:
        """Apply the z-normalization.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            The input data with shape :math:`(C, H, W, D)` or :math:`(H, W, D)`

        Returns
        -------
        array or torch.Tensor
            The z-normalized data per channel with same type as input.
        """

        is_torch = isinstance(data, torch.Tensor)

        def mean(x):
            return torch.mean(x) if is_torch else np.mean(x)

        def std(x):
            return torch.std(x) if is_torch else np.std(x)

        def cast_eps(ref):
            return (
                torch.tensor(self.eps, dtype=ref.dtype, device=ref.device)
                if is_torch
                else ref.dtype.type(self.eps)
            )

        output = (
            np.empty_like(data)
            if isinstance(data, np.ndarray)
            else torch.empty_like(data)
        )

        if data.ndim == 3:
            masked = (
                self.masking_fn(data)
                if self.masking_fn is not None
                else slice(None)
            )
            stats_data = data[masked] if self.masking_fn is not None else data
            mu = mean(stats_data)
            sigma = std(stats_data)
            eps_val = cast_eps(data)
            return (data - mu) / (sigma + eps_val)
        else:
            for c in range(data.shape[0]):
                stats_data = data[c]
                if self.masking_fn is not None:
                    stats_data = data[c][self.masking_fn(data[c])]
                mu = mean(stats_data)
                sigma = std(stats_data)
                eps_val = cast_eps(data)
                output[c] = (data[c] - mu) / (sigma + eps_val)
        return output
