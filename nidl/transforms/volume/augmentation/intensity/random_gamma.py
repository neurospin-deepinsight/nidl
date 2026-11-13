##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

import random

from ....transforms import TypeTransformInput, VolumeTransform
from ._utils import _like, _to_float_numpy


class RandomGamma(VolumeTransform):
    r"""Random gamma contrast change of a 3d volume:

    .. math::

        x' = \left(\frac{x - m}{M - m}\right)^{\gamma} (M - m) + m

    with :math:`m`, :math:`M` the volume min and max.

    Reimplementation of MONAI's ``RandAdjustContrast`` [1]_ (Apache-2.0).
    Non-linear, unlike ``RandomContrastAdjust`` (linear gain + bias).

    Parameters
    ----------
    gamma: (float, float), default=(0.5, 2.0)
        Range of :math:`\gamma \sim \mathcal{U}(a, b)`.
    kwargs: dict
        Keyword arguments.

    References
    ----------
    .. [1] Cardoso, M. J., et al., "MONAI: An open-source framework for deep
           learning in healthcare." arXiv, 2022.
           https://arxiv.org/abs/2211.02701
    """

    def __init__(self, gamma: tuple[float, float] = (0.5, 2.0), **kwargs):
        super().__init__(**kwargs)
        self.gamma = self._parse_range(gamma, check_min=0)

    def apply_transform(self, data: TypeTransformInput) -> TypeTransformInput:
        """Apply a random gamma transform.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            The input volume.

        Returns
        -------
        data: np.ndarray or torch.Tensor
            Transformed volume, same type and shape as the input.
        """
        gamma = random.uniform(*self.gamma)
        x = _to_float_numpy(data)
        lo, hi = x.min(), x.max()
        out = ((x - lo) / (hi - lo + 1e-7)) ** gamma * (hi - lo) + lo
        return _like(out, data)
