##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Intensity-based transformations.
"""

import itertools
from typing import ClassVar

import numpy as np
import torch

from ..base import (
    Ico,
)
from ..nn import (
    IcoConv,
)
from .utils import (
    Interval,
    validate_data,
)


class RandomGaussianBlur:
    """
    Add Gaussian blur to input data with random parameters.

    Parameters
    ----------
    sigma : tuple[float, float]
        Standard deviation of the Gaussian distribution. Range (low, high)
        from which the standard deviation is sampled uniformly.
        Default is (0.0, 1.0)
    standard_ico : bool
        If True, recursively subdivides a standard icosahedron.
        If False, loads precomputed FreeSurfer icosahedra.
        Default False.

    Attributes
    ----------
    sigma_ : list[int]
        Standard deviation of the Gaussian distribution.
    depth_ : int
        Estimated depth to build neighbors.

    Examples
    --------
    >>> from nidl.surface.transforms import RandomGaussianBlur
    >>> import numpy as np
    >>> transform = RandomGaussianBlur(sigma=(0.0, 1.0))
    >>> x = np.ones(642)
    >>> x_blured = transform(x)
    >>> x_blured.shape
    (642,)
    """
    bounds: ClassVar[dict[str, tuple[float | None, float | None, type]]] = {
        "sigma": (0, None, float),     # sigma must be >= 0
    }

    def __init__(
            self,
            sigma: tuple[float, float] = (0.0, 1.0),
            standard_ico: bool = False,
        ) -> None:
        self.sigma_sampler = Interval(sigma, self.bounds["sigma"])
        self.standard_ico = standard_ico
        self.sigma_ = None
        self.depth_ = None

    @validate_data(allowed_dims=(1, 2))
    def __call__(
            self,
            texture: np.ndarray,
        ) -> np.ndarray:
        """
        Add Gaussian blur to texture.

        Parameters
        ----------
        texture : np.ndarray, shape (N,) or (C, N)
            Input texture.

        Returns
        -------
        np.ndarray
            Blured texture (new array, not in-place).
        """
        self.sigma_ = self.sigma_sampler()

        # Determine order
        order = Ico.order(len(texture.T))
        self.depth_ = max(1, int(2 * self.sigma_sampler.high + 0.5))

        # Build ring index array
        positions = np.array(
            [0, *list(
                itertools.chain(
                    *[
                        [ring] * (6 * ring)
                        for ring in range(1, self.depth_ + 1)
                    ]
                )
            )]
        )

        # Gaussian kernel
        kernel = np.exp(-0.5 * (positions / self.sigma_) ** 2)
        kernel /= kernel.sum()

        # Apply convolution
        conv = IcoConv(
            1, 1,
            order=order,
            depth=self.depth_,
            standard_ico=self.standard_ico,
            bias=False,
        )
        with torch.no_grad():
            conv.weight.weight = torch.nn.Parameter(
                torch.Tensor(kernel), False
            )
        texture = conv(torch.from_numpy(texture[None]).float())

        return texture.numpy()[0]


class RandomGaussianNoise:
    """
    Add Gaussian noise to input data with random parameters.

    Parameters
    ----------
    mean : float | tuple[float, float]
        Mean of the Gaussian distribution. If a tuple (low, high) is provided,
        the mean is sampled uniformly from [low, high].
        Default is 0.
    sigma : tuple[float, float]
        Range (low, high) from which the standard deviation is sampled
        uniformly.
        Default is (0.01, 0.1).

    Attributes
    ----------
    mean_ : int
        Mean of the Gaussian distribution.
    sigma_ : list[int]
        Standard deviation of the Gaussian distribution.

    Examples
    --------
    >>> from nidl.surface.transforms import RandomGaussianNoise
    >>> import numpy as np
    >>> transform = RandomGaussianNoise(mean=(0, 0), sigma=(0.01, 0.1))
    >>> x = np.ones(642)
    >>> x_noised = transform(x)
    >>> x_noised.shape
    (642,)
    """
    bounds: ClassVar[dict[str, tuple[float | None, float | None, type]]] = {
        "mean": (None, None, float),   # no constraint
        "sigma": (0, None, float),     # sigma must be >= 0
    }

    def __init__(
            self,
            mean: float | tuple[float, float] = 0.0,
            sigma: tuple[float, float] = (0.01, 0.1),
        ) -> None:
        self.mean_sampler = Interval(mean, self.bounds["mean"])
        self.sigma_sampler = Interval(sigma, self.bounds["sigma"])
        self.mean_ = None
        self.sigma_ = None

    @validate_data(allowed_dims=(1, 2))
    def __call__(
            self,
            texture: np.ndarray,
        ) -> np.ndarray:
        """
        Add Gaussian noise to texture.

        Parameters
        ----------
        texture : np.ndarray, shape (N,) or (C, N)
            Input texture.

        Returns
        -------
        np.ndarray
            Noised texture (new array, not in-place).
        """
        self.mean_ = self.mean_sampler()
        self.sigma_ = self.sigma_sampler()
        noise = np.random.normal(self.mean_, self.sigma_, size=texture.shape)
        return texture + noise
