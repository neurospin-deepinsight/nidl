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
from typing import Optional, Union

from ....transforms import TypeTransformInput, VolumeTransform


class RandomContrastAdjust(VolumeTransform):
    r"""Randomly adjust the brightness and contrast of a 3d volume,
    using the linear transformation:

    .. math::

        p_\text{out} = \alpha \cdot p_\text{in} + \beta

    where :math:`p_\text{in}` (resp. :math:`p_\text{out}`) is the input
    (resp. output) voxel intensity, :math:`\alpha` is the contrast factor
    (gain) and :math:`\beta` is the brightness factor (bias). The same
    transform is applied across channels.

    Parameters
    ----------
    contrast_factor: (float, float), default=(0.75, 1.25)
        Contrast factor (gain) :math:`\alpha`, sampled
        :math:`\alpha \sim \mathcal{U}(a, b)`. A factor of 1.0 leaves the
        contrast unchanged, a factor in :math:`[0, 1)` reduces it and a factor
        greater than 1.0 increases it. Both bounds must be :math:`\ge 0`
        (a negative gain would invert intensities, not adjust contrast).
    brightness_factor: float or (float, float), default=0.0
        Brightness factor (bias) :math:`\beta`. If two values
        :math:`(a, b)` are given, then
        :math:`\beta \sim \mathcal{U}(a, b)`. If a single number ``b`` is
        given, the symmetric range ``(-b, b)`` is used. A factor of 0.0 leaves
        the brightness unchanged.
    output_range: (float, float) or None, default=None
        If a tuple :math:`(low, high)` is given, the output is clipped to
        this range to discard impossible intensity values introduced by the
        adjustment (e.g. ``(0, 1)`` for min-max rescaled inputs). If
        ``None`` (default), no clipping is performed.
    kwargs: dict
        Keyword arguments given to :class:`nidl.transforms.Transform`
        (e.g. ``p``, the probability of applying the transform).

    Notes
    -----
    This transformation can be used to simulate scanner variability. Note that
    this transform affects the mean if `brightness_factor` is not 0.

    Examples
    --------
    >>> import torch
    >>> from nidl.transforms.volume.augmentation.intensity import (
    ...     RandomContrastAdjust)
    >>> volume = torch.randn(1, 64, 64, 64)
    >>> transform = RandomContrastAdjust(
    ...     contrast_factor=(0.8, 1.2), brightness_factor=(-0.1, 0.1))
    >>> adjusted = transform(volume)  # shape (1, 64, 64, 64)
    """

    def __init__(
        self,
        contrast_factor: tuple[float, float] = (0.75, 1.25),
        brightness_factor: Union[float, tuple[float, float]] = 0.0,
        output_range: Optional[tuple[float, float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.contrast_factor = self._parse_range(contrast_factor, check_min=0)

        if isinstance(brightness_factor, numbers.Number):
            if brightness_factor < 0:
                raise ValueError(
                    "A single brightness_factor must be non-negative, got "
                    f"{brightness_factor}"
                )
            brightness_factor = (-brightness_factor, brightness_factor)
        self.brightness_factor = self._parse_range(brightness_factor)

        if output_range is not None:
            output_range = self._parse_range(output_range)
        self.output_range = output_range

    def apply_transform(self, data: TypeTransformInput) -> TypeTransformInput:
        """Adjust the brightness and contrast of the input.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            The input volume.

        Returns
        -------
        data: np.ndarray or torch.Tensor
            Brightness/contrast adjusted volume. Output type and shape are
            the same as input.
        """
        alpha = random.uniform(*self.contrast_factor)
        beta = random.uniform(*self.brightness_factor)

        adjusted = data * alpha + beta

        if self.output_range is not None:
            adjusted = adjusted.clip(*self.output_range)

        return adjusted
