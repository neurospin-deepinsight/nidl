##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import numbers
import random
from typing import Union

import numpy as np
import torch

from .....transforms import TypeTransformInput, VolumeTransform


class RandomGaussianNoise(VolumeTransform):
    """Add Gaussian noise to input data with random parameters.

    The input data can have any shape with type `np.ndarray` or
    `torch.Tensor`. The output has consistent type and shape with
    the input.

    Parameters
    ----------
    mean: float or (float, float), default=0.0
        Mean :math:`\\mu` of the Gaussian distribution from which the noise
        is sampled. If two values :math:`(a, b)` are given, then
        :math:`\\mu \\sim \\mathcal{U}(a, b)`.
    std: (float, float), default=(0.1, 1.0)
        Range of the standard deviation :math:`(a, b)` of the Gaussian
        distribution from which the noise is sampled
        :math:`\\sigma \\sim \\mathcal{U}(a, b)`.
    kwargs: dict
        Keyword arguments.
    """

    def __init__(
        self,
        mean: Union[float, tuple[float, float]] = 0.0,
        std: tuple[float, float] = (0.1, 1.0),
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(mean, numbers.Number):
            mean = (mean, mean)
        self.mean = self._parse_range(mean)
        self.std = self._parse_range(std, check_min=0)

    def apply_transform(self, data: TypeTransformInput) -> TypeTransformInput:
        """Add Gaussian noise to the input.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            The input volume.

        Returns
        -------
        data: np.ndarray or torch.Tensor
            Input with noise.
        """
        mean = random.uniform(*self.mean)
        std = random.uniform(*self.std)
        data_is_tensor = isinstance(data, torch.Tensor)
        if data_is_tensor:
            dtype, device = data.dtype, data.device
            data = data.detach().cpu().numpy()

        noise = np.random.normal(mean, std, size=data.shape).astype(data.dtype)
        noised_data = data + noise

        if data_is_tensor:
            noised_data = torch.as_tensor(
                noised_data, dtype=dtype, device=device
            )

        return noised_data
