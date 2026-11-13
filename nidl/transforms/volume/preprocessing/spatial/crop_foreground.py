##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

import numpy as np
import torch

from ....transforms import TypeTransformInput, VolumeTransform


class CropForeground(VolumeTransform):
    """Crop a 3d volume to the bounding box of its foreground
    (voxels above `threshold` in any channel).

    Reimplementation of MONAI's ``CropForeground`` [1]_ (Apache-2.0).

    Parameters
    ----------
    threshold: float, default=0.0
        Voxels strictly above this value are foreground.
    kwargs: dict
        Keyword arguments.

    References
    ----------
    .. [1] Cardoso, M. J., et al., "MONAI: An open-source framework for deep
           learning in healthcare." arXiv, 2022.
           https://arxiv.org/abs/2211.02701
    """

    def __init__(self, threshold: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def apply_transform(self, data: TypeTransformInput) -> TypeTransformInput:
        """Crop to the foreground (unchanged if there is none).

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            The input volume.

        Returns
        -------
        data: np.ndarray or torch.Tensor
            Cropped volume, same type as the input.
        """
        mask = data > self.threshold
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if mask.ndim == 4:
            mask = mask.any(axis=0)
        if not mask.any():
            return data
        idx = np.argwhere(mask)
        lo, hi = idx.min(axis=0), idx.max(axis=0) + 1
        return data[..., lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
