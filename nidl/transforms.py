##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""This modules details the public API you should use and implement for a
nidl compatible transform, as well as the transforms available in nidl.
"""

from __future__ import annotations

import numbers
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import numpy as np
import torch

TypeTransformInput = Union[np.ndarray, torch.Tensor]


class Transform(ABC):
    """Abstract class for all nidl transformations.

    The general logic for any transformation when called is:

    1) calling to  :meth:`parse_data` for parsing the input data, a
       :class:`numpy.ndarray` or :class:`torch.Tensor`, and validate
       its dimension. The output should be formatted with verified shape
       and type.

    2) (optional) calling to :func:`random.random` to know whether the
       transformation is applied or not (depending on a probability :math:`p`).

    3) calling to :meth:`apply_transform` for applying the transformation on
       the formatted data. This abstract method should be implemented in every
       subclass to specify the actual transformation.

    Transformations in nidl are compliant with `torchvision.transforms`
    module and it can be used in conjonction.

    Spatial augmentation currently implemented (change geometry):
        - RandomErasing (3d)
        - RandomResizedCrop (3d)
        - RandomFlip (3d-array)
        - RandomRotation (3d)

    Intensity augmentations currently implemented (change voxel values):
        - RandomGaussianBlur (3d)
        - RandomGaussianNoise (nd)
        - Gamma (TODO)
        - RandomBrightness (TODO)
        - Biasfield (TODO)

    Intensity preprocessing:
        - ZNormalization (3d)
        - RobustRescaling (3d)

    Spatial preprocessing:
        - Resample (3d)
        - Resize (3d)
        - CropOrPad (3d)


    Parameters
    ----------
    p: float or None, default=None
        Probability to apply the transformation.
        If float, it should be between 0 and 1 (included).
        If None (default), the transformation is always applied.
    """

    def __init__(self, p: Union[float, None] = None):
        self.p = self.parse_probability(p)

    def __call__(self, data: TypeTransformInput, *args, **kwargs) -> Any:
        """Transform the input data.

        Parameters
        ----------
        data: TypeTransformInput
            Input data (usually :class:`numpy.ndarray` or :class:`torch.Tensor`
            to be transformed.
        *args: Any
            Additional positional arguments given to :meth:`apply_transform`.
        **kwargs: dict
            Additional keyword arguments given to :meth:`apply_transform`.

        Returns
        -------
        data_transformed: Any
            Transformed data.
        """
        data = self.parse_data(data)

        if self.p is not None and random.random() > self.p:
            return data

        with np.errstate(all="raise", under="ignore"):
            data_transformed = self.apply_transform(data, *args, **kwargs)

        return data_transformed

    def parse_data(self, data: TypeTransformInput) -> Any:
        """Parse the input data and returns formatted data.

        Input data must be a :class:`numpy.ndarray` or :class:`torch.Tensor`.

        Parameters
        ----------
        data: TypeTransformInput
            Input data to be transformed.

        Returns
        -------
        data: Any
            The formatted data.

        Raises
        ------
        ValueError
            If the input data is not :class:`numpy.ndarray` or
            :class:`torch.Tensor`
        """
        if not isinstance(data, (torch.Tensor, np.ndarray)):
            raise TypeError(
                f"Unexpected input type: {type(data)}, should be torch.Tensor "
                "or np.ndarray"
            )

        return data

    @abstractmethod
    def apply_transform(self, data_parsed: Any, *args, **kwargs) -> Any:
        """Apply the transformation on the data parsed by :meth:`parse_data`.
        This should be implemented in all subclasses.

        Parameters
        ----------
        data_parsed: Any
            Input data with type and shape already checked.
        *args: Any
            Additional positional arguments.
        **kwargs: dict
            Additional keyword arguments.

        Returns
        -------
        data: Any
            The transformed data.
        """
        raise NotImplementedError

    @staticmethod
    def parse_probability(probability: Union[float, None]) -> float:
        """Check if the probability is correct.

        In details, it checks whether it is a scalar (int or float) between
        0 and 1 (included).

        Parameters
        ----------
        probability: float or None
            Probability to check. None value is accepted.

        Raises
        ------
        ValueError
            If probability value is not supported.
        """
        if probability is None:
            return probability

        is_number = isinstance(probability, numbers.Number)
        if not (is_number and 0 <= probability <= 1):
            message = (
                f"Probability must be a number in [0, 1], not {probability}"
            )
            raise ValueError(message)
        return probability

    @property
    def name(self):
        return self.__class__.__name__

    @staticmethod
    def _parse_range(interval, check_min=None, check_max=None):
        """Checks if the input interval is correct.

        In details, it checks whether it contains two values :math:`(l, u)`
        such that :math:`l` and :math:`u` are scalar (int or float) and
        :math:`l \\le u`. Optionally, it also checks that
        :math:`l \\ge \\text{check_min}` and :math:`u \\le \\text{check_max}`.

        Parameters
        ----------
        interval: (float, float)
            The interval to check.
        check_min: float or None, default=None
            If float, check that lower bound of `interval` is superior to
            this (inclusive).
        check_max: float or None, default=None
            If float, check that upper bound of `interval` is inferior to
            this (inclusive).

        Returns
        -------
        interval: (float, float)
            The checked interval as tuple of float.

        Raises
        ------
        ValueError
            If interval is incorrect.
        """
        if not isinstance(interval, tuple):
            interval = tuple(interval)

        if len(interval) != 2:
            raise ValueError(
                f"Input interval must have size 2, got {len(interval)}"
            )
        if not isinstance(interval[0], numbers.Number) or not isinstance(
            interval[1], numbers.Number
        ):
            raise TypeError(
                f"Input interval must contain scalars, got {interval}"
            )
        if interval[0] > interval[1]:
            raise ValueError(
                f"Input interval must be s.t. lower <= upper, got {interval}"
            )
        if check_min is not None and interval[0] < check_min:
            raise ValueError(
                f"Lower bound must be >= {check_min}, got {interval[0]}"
            )
        if check_max is not None and interval[1] > check_max:
            raise ValueError(
                f"Lower bound must be <= {check_max}, got {interval[1]}"
            )
        return interval

    @staticmethod
    def _parse_shape(
        shape: Union[int, tuple[int, ...]], length: Optional[int] = None
    ) -> tuple[int, ...]:
        """Checks if the input shape is correct.

        In details, it checks if each dimension is a positive integer. If the
        length is specified, it also check if the input shape has correct
        length. If a single integer is given, it returns a tuple with length
        specified by `length` (default is 1 if not specified).

        Parameters
        ----------
        shape: int or tuple of int
            The shape to check.
        length: int or None, default=None
            The length of shape (optional).

        Returns
        -------
        shape: tuple of int
            The checked shape as tuple of int.

        Raises
        ------
        ValueError
            If shape has incorrect format.
        """

        def check_shape(s):
            if not isinstance(s, int) or s < 0:
                raise ValueError(
                    f"`shape` must contain positive int, got {shape}"
                )

        if isinstance(shape, int):
            check_shape(shape)
            if length is not None:
                return length * (shape,)
            return (shape,)
        elif isinstance(shape, (tuple, list)):
            shape = tuple(shape)
            if length is not None and len(shape) != length:
                raise ValueError(
                    f"`shape` must have {length} dimensions, got {len(shape)}"
                )
            for s in shape:
                check_shape(s)
            return shape
        raise ValueError(
            f"`shape` must be int, list or tuple of int, got {shape}"
        )


class Identity(Transform):
    """Identity transformation.

    It parses the input data (checking its type) and outputs the same data.
    """

    def apply_transform(self, data_parsed: Any, **kwargs):
        """
        Parameters
        ----------
        data_parsed: Any
            Input data with type checked.
        kwargs: dict
            Additional keyword arguments. Ignored.

        Returns
        -------
        data: Any
            Same as input.
        """
        return data_parsed


class MultiViewsTransform(Transform):
    """Multi-views transformation.

    It generates several "views" of the same input data, i.e. it applies
    transformations (usually stochastic) multiple times to the input.

    Parameters
    ----------
    transforms: Callable or Sequence of Callable
        Transformation or sequence of transformations to be applied.
        If a single transform is given, it generates `n_views` of the
        same input using the same transformation applied `n_views` times.
        If a sequence is given, it applies this sequence of transforms to
        the input in the same order.
    n_views: int or None, default=None
        Number of views to generate if `transforms` is a Transform.
        If n_views != 1 and `transforms` is a sequence, a ValueError
        is raised.
        If None, it is set to 1 if  `transforms` is a Transform and ignored
        otherwise.
    kwargs: dict
        Additional keyword arguments given to Transform.

    Returns
    -------
    data: tuple of array or torch.Tensor
        Tuple of transformed data.

    Notes
    -----
    The data are not parsed by this transformation. It should be handled
    elsewhere.
    """

    def __init__(
        self,
        transforms: Union[Callable, Sequence[Callable]],
        n_views: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_views = self._parse_nviews(n_views, transforms)

        self.transforms = []

        if callable(transforms):
            for _ in range(self.n_views):
                self.transforms.append(transforms)
        elif isinstance(transforms, Sequence):
            for transform in transforms:
                if not callable(transform):
                    message = (
                        "One or more transform(s) are not callable: "
                        f"got {type(transform)}"
                    )
                    raise TypeError(message)
                self.transforms.append(transform)
        else:
            raise TypeError(
                f"Unexpected transforms, got {type(transforms)} but expected "
                "a callable or sequence of callable"
            )

    def parse_data(self, data: Any):
        """Data are not parsed here."""
        return data

    @staticmethod
    def _parse_nviews(n_views, transforms):
        if n_views is None:
            n_views = 1

        if not isinstance(n_views, int):
            raise TypeError(
                f"n_views should be None or int, got {type(n_views)}"
            )
        if isinstance(transforms, Sequence) and n_views != 1:
            raise ValueError(
                "n_views != 1 and a sequence of transforms is given."
            )
        if n_views < 0:
            raise ValueError("n_views must be positive")
        return n_views

    def apply_transform(self, x: Any, **kwargs) -> tuple[TypeTransformInput]:
        return tuple(transform(x, **kwargs) for transform in self.transforms)


class VolumeTransform(Transform):
    """Transformation applied to a 3d volume."""

    def parse_data(self, data: TypeTransformInput) -> TypeTransformInput:
        """Checks if the input data shape is 3d or 4d.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            The input data with shape :math:`(C, H, W, D)` or
            :math:`(H, W, D)`

        Returns
        -------
        np.ndarray or torch.Tensor
            Data with type and shape checked.

        Raises
        ------
        ValueError
            If the input data is not :class:`numpy.ndarray` or
            :class:`torch.Tensor` or if the shape is not 3d or 4d.
        """
        data = super().parse_data(data)

        if len(data.shape) not in [3, 4]:
            raise ValueError(
                "Input data must be 3d or 4d (channel+spatial dimensions), "
                f"got {len(data.shape)}"
            )
        return data
