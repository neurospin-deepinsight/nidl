##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

import random

import numpy as np

from ....transforms import TypeTransformInput, VolumeTransform
from ._utils import _like, _to_float_numpy


class RandomGibbsNoise(VolumeTransform):
    r"""Random Gibbs ringing (MRI artifact) in a 3d volume.

    Keeps the k-space frequencies within a sphere of radius
    :math:`(1 - \alpha) \frac{\sqrt{2}}{2} \max(H, W, D)`. As in MONAI, for
    even sizes and :math:`\alpha` close to 1 the output can be 0.
    Reimplementation of MONAI's ``RandGibbsNoise`` [1]_ (Apache-2.0).

    Parameters
    ----------
    alpha: (float, float), default=(0.0, 1.0)
        Range of :math:`\alpha`, within :math:`[0, 1]`.
    kwargs: dict
        Keyword arguments.

    References
    ----------
    .. [1] Cardoso, M. J., et al., "MONAI: An open-source framework for deep
           learning in healthcare." arXiv, 2022.
           https://arxiv.org/abs/2211.02701
    """

    def __init__(self, alpha: tuple[float, float] = (0.0, 1.0), **kwargs):
        super().__init__(**kwargs)
        self.alpha = self._parse_range(alpha, check_min=0, check_max=1)

    def apply_transform(self, data: TypeTransformInput) -> TypeTransformInput:
        """Add Gibbs ringing.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            The input volume.

        Returns
        -------
        data: np.ndarray or torch.Tensor
            Volume with ringing, same type and shape as the input.
        """
        alpha = random.uniform(*self.alpha)
        x = _to_float_numpy(data)
        axes = (-3, -2, -1)
        k = np.fft.fftshift(np.fft.fftn(x, axes=axes), axes=axes)
        shape = np.array(x.shape[-3:])
        radius = (1 - alpha) * shape.max() * np.sqrt(2) / 2
        grid = np.ogrid[tuple(slice(0, n) for n in shape)]
        center = (shape - 1) / 2
        dist2 = sum((g - c) ** 2 for g, c in zip(grid, center, strict=False))
        k = k * (dist2 <= radius**2)  # same mask for every channel
        out = np.fft.ifftn(np.fft.ifftshift(k, axes=axes), axes=axes).real
        return _like(out, data)
