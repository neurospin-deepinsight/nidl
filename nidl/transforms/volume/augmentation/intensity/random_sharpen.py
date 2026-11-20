##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

import numbers
import random
from typing import Union

from scipy.ndimage import gaussian_filter

from ....transforms import TypeTransformInput, VolumeTransform
from ._utils import _like, _to_float_numpy


class RandomGaussianSharpen(VolumeTransform):
    r"""Random unsharp masking of a 3d volume:

    .. math::

        b_1 = G_{\sigma_1} * x, \quad b_2 = G_{\sigma_2} * b_1, \quad
        x' = b_1 + \alpha (b_1 - b_2)

    with :math:`G_\sigma` a Gaussian filter (one :math:`\sigma` per axis).

    Reimplementation of MONAI's ``RandGaussianSharpen`` [1]_ (Apache-2.0),
    with scipy's Gaussian filter (reflected borders, as
    :class:`RandomGaussianBlur`).

    Parameters
    ----------
    sigma1: (float, float), default=(0.5, 1.0)
        Range of :math:`\sigma_1`.
    sigma2: float or (float, float), default=0.5
        Range of :math:`\sigma_2`; a float :math:`c` means
        :math:`(c, \sigma_1)`.
    alpha: (float, float), default=(10.0, 30.0)
        Range of :math:`\alpha`.
    kwargs: dict
        Keyword arguments.

    References
    ----------
    .. [1] Cardoso, M. J., et al., "MONAI: An open-source framework for deep
           learning in healthcare." arXiv, 2022.
           https://arxiv.org/abs/2211.02701
    """

    def __init__(
        self,
        sigma1: tuple[float, float] = (0.5, 1.0),
        sigma2: Union[float, tuple[float, float]] = 0.5,
        alpha: tuple[float, float] = (10.0, 30.0),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sigma1 = self._parse_range(sigma1, check_min=0)
        if isinstance(sigma2, numbers.Number):
            if sigma2 < 0 or sigma2 > self.sigma1[0]:
                raise ValueError(
                    f"`sigma2` must be in [0, {self.sigma1[0]}], got {sigma2}"
                )
            self.sigma2 = sigma2
        else:
            self.sigma2 = self._parse_range(sigma2, check_min=0)
        self.alpha = self._parse_range(alpha, check_min=0)

    def apply_transform(self, data: TypeTransformInput) -> TypeTransformInput:
        """Sharpen the volume.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            The input volume.

        Returns
        -------
        data: np.ndarray or torch.Tensor
            Sharpened volume, same type and shape as the input.
        """
        sigma1 = [random.uniform(*self.sigma1) for _ in range(3)]
        if isinstance(self.sigma2, tuple):
            sigma2 = [random.uniform(*self.sigma2) for _ in range(3)]
        else:
            sigma2 = [random.uniform(self.sigma2, s) for s in sigma1]
        alpha = random.uniform(*self.alpha)
        x = _to_float_numpy(data)
        channel = [0] * (x.ndim - 3)  # no filtering across channels
        b1 = gaussian_filter(x, channel + sigma1)
        b2 = gaussian_filter(b1, channel + sigma2)
        return _like(b1 + alpha * (b1 - b2), data)
