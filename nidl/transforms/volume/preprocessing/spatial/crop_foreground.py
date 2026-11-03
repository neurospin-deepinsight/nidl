##########################################################################
# NSAp - Copyright (C) CEA, 2026
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
    """Crop a 3d volume around its foreground.

    The foreground is made of the voxels strictly above `threshold` in any
    channel. The bounding box is enlarged by `margin` voxels on each side,
    without going beyond the volume.

    It handles :class:`numpy.ndarray` or :class:`torch.Tensor` as input and
    returns a consistent output (same type).

    Parameters
    ----------
    margin : int, default=0
        Number of voxels added around the foreground on each side.
    threshold : float, default=0.0
        Voxels strictly above this value are in the foreground.
    kwargs : dict
        Keyword arguments given to
        :class:`nidl.transforms.transforms.Transform`.

    Examples
    --------
    >>> import numpy as np
    >>> from nidl.transforms.volume.preprocessing import CropForeground
    >>> volume = np.zeros((30, 30, 30))
    >>> volume[10:12, 1:3, 25:27] = 1
    >>> CropForeground(margin=4)(volume).shape
    (10, 7, 9)
    """

    def __init__(self, margin: int = 0, threshold: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        if margin < 0:
            raise ValueError(f"margin must be positive, got {margin}.")
        self.margin = margin
        self.threshold = threshold

    def apply_transform(self, data: TypeTransformInput) -> TypeTransformInput:
        """Crop the input data around its foreground.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            The input data with shape :math:`(C, H, W, D)` or
            :math:`(H, W, D)`.

        Returns
        -------
        data: np.ndarray or torch.Tensor
            Cropped data with same type as input.

        Raises
        ------
        ValueError
            If the volume has no foreground.
        """
        foreground = data > self.threshold
        if isinstance(foreground, torch.Tensor):
            foreground = foreground.cpu().numpy()
        if data.ndim == 4:
            foreground = foreground.any(axis=0)
        if not foreground.any():
            raise ValueError("Cannot crop a volume without foreground.")
        bounding_box = [slice(None)] * (data.ndim - 3)
        for axis in range(3):
            other_axes = tuple(i for i in range(3) if i != axis)
            indices = np.flatnonzero(foreground.any(axis=other_axes))
            start = max(indices[0] - self.margin, 0)
            end = min(indices[-1] + self.margin + 1, foreground.shape[axis])
            bounding_box.append(slice(start, end))
        return data[tuple(bounding_box)]
