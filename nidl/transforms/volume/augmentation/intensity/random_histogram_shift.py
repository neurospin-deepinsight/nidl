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


class RandomHistogramShift(VolumeTransform):
    """Remap the intensities of a 3d volume with a random increasing curve.

    The curve interpolates control points evenly spaced on the intensity
    range, with the inner ones moved at random (fewer points, larger
    shifts). Reimplementation of MONAI's ``RandHistogramShift`` [1]_
    (Apache-2.0).

    Parameters
    ----------
    num_control_points: int, default=10
        Number of control points (at least 3).
    kwargs: dict
        Keyword arguments.

    References
    ----------
    .. [1] Cardoso, M. J., et al., "MONAI: An open-source framework for deep
           learning in healthcare." arXiv, 2022.
           https://arxiv.org/abs/2211.02701
    """

    def __init__(self, num_control_points: int = 10, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(num_control_points, int) or num_control_points < 3:
            raise ValueError(
                "`num_control_points` must be an integer >= 3, got "
                f"{num_control_points}"
            )
        self.num_control_points = num_control_points

    def apply_transform(self, data: TypeTransformInput) -> TypeTransformInput:
        """Remap the intensities (a constant volume is unchanged).

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            The input volume.

        Returns
        -------
        data: np.ndarray or torch.Tensor
            Remapped volume, same type and shape as the input.
        """
        reference = np.linspace(0, 1, self.num_control_points)
        floating = reference.copy()
        for i in range(1, self.num_control_points - 1):
            floating[i] = random.uniform(floating[i - 1], floating[i + 1])
        x = _to_float_numpy(data)
        lo, hi = x.min(), x.max()
        if hi == lo:
            return data
        out = np.interp(x, lo + reference * (hi - lo),
                        lo + floating * (hi - lo))
        return _like(out, data)
