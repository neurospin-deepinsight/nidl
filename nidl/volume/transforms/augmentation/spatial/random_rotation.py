##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import random
from typing import Optional, Union

import numpy as np
import torch
from nibabel.orientations import aff2axcodes

from ...volume_transform import TypeTransformInput, VolumeTransform


class RandomRotation(VolumeTransform):
    """Randomly rotates a 3d volume by 90-degree multiples around spatial axes.

    Parameters
    ----------
    axes : (int or str, int or str), or tuple of (int or str, int or str),\
        default=(0, 1)
        2d axis tuple to potentially rotate over. Anatomical labels could be
        used as well such as "LR" (Left-Right), "AP" (Antero-Posterior) or
        "IS" (Inferior-Posterior). In that case, RAS-formated affine matrix is
        required when the transformation is called. If a tuple of pairs is
        given, multiple rotations are eventually applied around each plane.
    rotation_probability : float, default=1.0
        Probability to apply rotation for each axis pair.
    kwargs : dict
        Keyword arguments.
    
    Examples
    --------
    >>> import torch
    >>> from nidl.volume.transforms.augmentation.spatial import RandomRotation
    >>> volume = torch.randn(1, 64, 64, 64)  # shape: (C, H, W, D)
    >>> transform = RandomRotation(axes=("LR", "AP"), rotation_probability=0.5)
    >>> rotated = transform(volume) # shape (1, 64, 64, 64)
    
    """

    def __init__(
        self,
        axes: Union[
            tuple[Union[int, str], Union[int, str]],
            tuple[tuple[Union[int, str], Union[int, str]], ...],
        ] = (0, 1),
        rotation_probability: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.axes = self._parse_axes(axes)
        self.rotation_probability = self.parse_probability(
            rotation_probability
        )

    def apply_transform(
        self,
        data: TypeTransformInput,
        affine: Optional[np.ndarray] = None,
    ) -> TypeTransformInput:
        """Apply random 90-degree rotations.

        Parameters
        ----------
        data : np.ndarray or torch.Tensor
            Input volume of shape :math:`(C, H, W, D)` or :math:`(H, W, D)`.

        affine: np.ndarray of shape (4, 4) or None, default=None
            Affine transformation matrix of the input data in RAS format
            defining orientation of the input image. This is typically given
            by Nibabel in this format. This is used only if anatomical labels
            are provided in the flipped axes ("LR", "AP" or "IS") and ignored
            otherwise. If None, the identity matrix is used, assuming RAS
            orientation.

        Returns
        -------
        Rotated volume of same type and shape as input.
        """
        affine = np.eye(4) if affine is None else affine
        for axis_pair in self.axes:
            if random.random() < self.rotation_probability:
                axes = self._translate_axes(axis_pair, affine)
                k = random.randint(1, 3)  # 90, 180, or 270 degrees
                data = self._rotate(data, k=k, axes=axes)

        return data

    def _rotate(self, data, k, axes):
        if data.ndim == 4:  # never rotate along channel dimension.
            axes = tuple(np.array(axes) + 1)
        if isinstance(data, torch.Tensor):
            data = torch.rot90(data, k=k, dims=axes)
        else:
            data = np.rot90(data, k=k, axes=axes)
        return data

    def _translate_axes(
        self,
        axis_pair: tuple[Union[int, str], Union[int, str]],
        affine: np.ndarray,
    ):
        converted_pair = []
        for a in axis_pair:
            if isinstance(a, int):
                converted_pair.append(a)
            else:  # string
                converted_pair.append(
                    self.get_index_from_anat_label(a, affine)
                )
        return tuple(converted_pair)

    def get_index_from_anat_label(self, axis: str, affine: np.ndarray) -> int:
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

    def _parse_axes(self, axes):
        axes = tuple(axes)
        if len(axes) == 2 and isinstance(axes[0], (int, str)):
            axes = (axes,)

        for pair in axes:
            if len(pair) != 2:
                raise ValueError(
                    "Each rotation axis pair must have two elements,"
                    f" got {pair}"
                )
            for a in pair:
                if not (isinstance(a, int) and a in (0, 1, 2)) and not (
                    isinstance(a, str) and a in ("LR", "AP", "IS")
                ):
                    raise ValueError(
                        f"Invalid axis {a}. Must be int in (0, 1, 2) or"
                        " anatomical label in ('LR'', 'AP', 'IS')."
                    )
        return axes
