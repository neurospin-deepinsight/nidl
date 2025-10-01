##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import random
from typing import Union

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .....transforms import TypeTransformInput, VolumeTransform


class RandomGaussianBlur(VolumeTransform):
    """Blur a 3d volume using a Gaussian filter with random kernel size.

    It handles a `np.ndarray` or `torch.Tensor` as input and
    returns a consistent output (same type and shape). Input shape must be
    :math:`(C, H, W, D)` or :math:`(H, W, D)` (spatial dimensions).

    Parameters
    ----------
    sigma: (float, float) or (float, float, float, float, float, float),\
        default=(0, 2)
        Range of the standard deviation :math:`\\sigma` of the Gaussian kernel
        applied to blur the volume.
        If two values :math:`(a,b)` are provided, then
        :math:`\\sigma \\sim \\mathcal{U}(a, b)`.
        If six values :math:`(a_1, b_1, a_2, b_2, a_3, b_3)` are provided, then
        one standard deviation per spatial dimension is sampled
        :math:`\\sigma_i \\sim \\mathcal{U}(a_i, b_i)` for :math:`i=1,2,3`.
    kwargs: dict
           Keyword arguments.
    """

    def __init__(
        self,
        sigma: Union[
            tuple[float, float],
            tuple[float, float, float, float, float, float],
        ] = (0, 2),
        **kwargs,
    ):
        """ """
        super().__init__(**kwargs)
        self.sigma = self._parse_range(
            sigma, check_min=0, check_length=6, name="sigma"
        )

    def apply_transform(self, data: TypeTransformInput) -> TypeTransformInput:
        """Blur the input with a Gaussian filter.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            Input volume with shape :math:`(C, H, W, D)` or :math:`(H, W, D)`.
            Standard deviations in the Gaussian filter are equal across
            channels.

        Returns
        -------
        data: np.ndarray or torch.Tensor
            Blurred volume. Output type and shape are the same as input.

        """

        def stack(list_data, data, dim=0):
            if isinstance(data, torch.Tensor):
                return torch.stack(list_data, dim=dim)
            return np.stack(list_data, axis=dim)

        def generic_gaussian_filter(data, std):
            if isinstance(data, torch.Tensor):
                blurred_data = gaussian_filter(
                    data.detach().cpu().numpy(), std
                )
                return torch.as_tensor(
                    blurred_data, dtype=data.dtype, device=data.device
                )
            return gaussian_filter(data, std)

        std = self._sample_uniform_params(self.sigma)
        if data.ndim == 4:  # (c, h, w, d)
            C = data.shape[0]
            return stack(
                [generic_gaussian_filter(data[c], std) for c in range(C)],
                data=data,
                dim=0,
            )
        else:
            return generic_gaussian_filter(data, std)

    @staticmethod
    def _sample_uniform_params(params):
        results = []
        for a, b in zip(params[::2], params[1::2]):
            results.append(random.uniform(a, b))
        return tuple(results)

    @staticmethod
    def _parse_range(
        intervals, check_min=None, check_max=None, check_length=None, name=None
    ):
        """Checks if the input interval(s) is correct.

        It handles arbitrary number of input intervals as long
        as it has even length.

        Parameters
        ----------
        intervals: tuple of float
            The interval(s) to check, typically
            :math:`(a_1, b_1, a_2, b_2, ...)`. The length should be even.

        check_min: float or None, default=None
            If float, check that lower bound of `intervals` is superior to
            this (inclusive). It checks all intervals.

        check_max: float or None, default=None
            If float, check that upper bound of `intervals` is inferior to
            this (inclusive). It checks all intervals.

        check_length: int or None, default=None
            If int, check that length of `intervals` is equal to this.
            If `intervals` is a pair, it is repeated to match the given
            length.

        name: str or None, default=None
            Name to display when raising errors.

        Returns
        -------
        interval: tuple of float
            The checked interval(s) as tuple of float.

        """
        name = "`intervals`" if name is None else f"`{name}"
        if not isinstance(intervals, tuple):
            intervals = tuple(intervals)

        if len(intervals) % 2 != 0:
            raise ValueError(
                f"{name} should have even length, got {len(intervals)}"
            )
        if check_length is not None and len(intervals) == 2:
            intervals = (check_length // 2) * intervals
        if check_length is not None and len(intervals) != check_length:
            raise ValueError(
                f"{name} must have {check_length} dimensions, got"
                f"{len(intervals)}"
            )
        for a, b in zip(intervals[::2], intervals[1::2]):
            VolumeTransform._parse_range(
                (a, b), check_min=check_min, check_max=check_max
            )
        return intervals
