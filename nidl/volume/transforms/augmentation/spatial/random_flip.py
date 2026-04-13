##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

import random
from typing import Optional, Union

import numpy as np
import torch
from nibabel.orientations import aff2axcodes

from .....transforms import TypeTransformInput, VolumeTransform


class RandomFlip(VolumeTransform):
    """Reverse the order of elements in a 3d volume along the given axes.

    It handles a :class:`numpy.ndarray` or :class:`torch.Tensor` as input and
    returns a consistent output (same type and shape). Input shape must be
    :math:`(C, H, W, D)` or :math:`(H, W, D)` (spatial dimensions).

    Parameters
    ----------
    axes: int, str or tuple of int or str, default=0
        Index in (0, 1, 2) or tuple of indices of the spatial dimension along
        which the input volume might be flipped. Anatomical labels could be
        used as well such as "LR" (Left-Right), "AP" (Antero-Posterior) or "IS"
        (Inferior-Posterior). In that case, RAS-formatted affine matrix
        specifying volume orientation must be provided when the transformation
        is called. Check `Nibabel documentation on image orientation
        <https://nipy.org/nibabel/coordinate_systems.html>`_.
    flip_probability: float, default=1.0
        Per-axis probability to flip the volume.
    kwargs: dict
        Keyword arguments.

    Notes
    -----
    Current implementation always returns a new tensor/array without sharing
    memory with the input data.

    """

    def __init__(
        self,
        axes: Union[int, str, tuple[Union[int, str], ...]] = 0,
        flip_probability: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.axes = self._parse_axes(axes)
        self.flip_probability = self.parse_probability(flip_probability)

    def apply_transform(
        self,
        data: TypeTransformInput,
        affine: Optional[np.ndarray] = None,
    ) -> TypeTransformInput:
        """Flip the volume along random axes.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            Input volume with shape :math:`(C, H, W, D)` or :math:`(H, W, D)`.
            Channel dimension will never be flipped.

        affine: np.ndarray of shape (4, 4) or None, default=None
            Affine transformation matrix of the input data in RAS format
            defining orientation of the input image. This is typically given
            by Nibabel in this format. This is used only if anatomical labels
            are provided in the flipped axes ("LR", "AP" or "IS") and ignored
            otherwise.
            If None, the identity matrix is used, assuming RAS orientation.

        Returns
        -------
        data: np.ndarray or torch.Tensor
            Flipped volume with same type and shape as the input.
        """

        flipped_axis = []
        for axis in self.axes:
            if self.flip_probability > random.random():
                if isinstance(axis, str):
                    flipped_axis.append(
                        self.get_index_from_anat_label(axis, affine)
                    )
                else:
                    flipped_axis.append(axis)

        if len(flipped_axis) > 0:
            if data.ndim == 4:
                flipped_axis = tuple(np.array(flipped_axis) + 1)
            if isinstance(data, torch.Tensor):
                data = torch.flip(data, dims=flipped_axis)
            else:  # copy data for consistency between NumPy and Torch
                data = np.flip(data, axis=flipped_axis).copy()
        return data

    def get_index_from_anat_label(self, axis: str, affine: np.ndarray):
        """Returns the axis index corresponding to a given anatomical label.

        Parameters
        ----------
        axis : {'LR', 'AP', 'IS'}
            Anatomical axis label:

            - 'LR' for Left-Right (X axis)
            - 'AP' for Anterior-Posterior (Y axis)
            - 'IS' for Inferior-Superior (Z axis)

        affine : np.ndarray
            4x4 affine matrix defining the orientation of the volume.

        Returns
        -------
        int
            The index (0, 1, or 2) in voxel space corresponding to the
            requested anatomical axis.
        """
        anat_to_physical = {
            "LR": ("L", "R"),
            "AP": ("P", "A"),
            "IS": ("I", "S"),
        }

        if axis not in anat_to_physical:
            raise ValueError(
                f"Invalid axis '{axis}'. Must be one of 'LR', 'AP', 'IS'."
            )

        desired = anat_to_physical[axis]
        axcodes = aff2axcodes(affine)

        for i, code in enumerate(axcodes):
            if code in desired:
                return i

        raise ValueError(
            f"Could not find anatomical axis '{axis}' in affine matrix."
        )

    def _parse_axes(self, axes: Union[int, tuple[int, ...], str]):
        if isinstance(axes, (int, str)):
            axes = (axes,)
        axes_tuple = tuple(axes)
        for axis in axes_tuple:
            if isinstance(axis, int):
                if axis not in (0, 1, 2):
                    raise ValueError(f"Axes must be 0, 1 or 2, got {axis}")
            elif isinstance(axis, str):
                if axis not in ("LR", "AP", "IS"):
                    raise ValueError(
                        f"Axes must be in 'LR', 'AP' or 'IS', got {axis}"
                    )
            else:
                raise TypeError(f"Axes must be int or str, got {axis}")
        return axes_tuple
