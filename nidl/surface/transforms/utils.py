##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Surfacic transform utilities.
"""

from functools import wraps
from typing import Callable

import numpy as np
import torch


class Interval:
    """
    Represents either a fixed value or a uniform sampling interval.

    - If `value` is a scalar then always return that value.
    - If `value` is (low, high) then sample uniformly.
    - If both bounds are integers then sample with randint.
    - Bounds are validated if provided.

    Parameters
    ----------
    value : float | tuple[float, float]
        Either a single value or a (low, high) interval.
    bounds : tuple[float | None, float | None]
        Optional (min, max) allowed values.
        Use None for no constraint.

    Raises
    ------
    ValueError
        If the specified value do not respect the provided bounds.
    """
    def __init__(
            self,
            value: float | tuple[float, float],
            bounds: tuple[float | None, float | None] = (None, None),
        ) -> None:
        # Normalize to (low, high)
        if isinstance(value, tuple):
            low, high = value
        else:
            low = high = value

        # Validate against bounds
        bmin, bmax, self.dtype = bounds
        if bmin is not None and low < bmin:
            raise ValueError(f"Value {low} < lower bound {bmin}")
        if bmin is not None and high < bmin:
            raise ValueError(f"Value {high} < lower bound {bmin}")
        if bmax is not None and low > bmax:
            raise ValueError(f"Value {low} > upper bound {bmax}")
        if bmax is not None and high > bmax:
            raise ValueError(f"Value {high} > upper bound {bmax}")
        if low > high:
            raise ValueError(f"Invalid interval: low={low} > high={high}")

        self.low = low
        self.high = high

    def __call__(self) -> float:
        """
        Sample a value from the interval.

        Raises
        ------
        ValueError
            If the data type is not supported.
        """
        if self.low == self.high:
            return self.low

        if self.dtype == int:
            return np.random.randint(self.low, self.high + 1)
        elif self.dtype == float:
            return np.random.uniform(self.low, self.high)
        else:
            raise ValueError(
                f"Data type not supported: {self.dtype}"
            )


def validate_data(allowed_dims: int | list[int]) -> Callable:
    """
    Decorator that validates the dimensionality and type of an input data
    argument.

    Works for both standalone functions and class methods.

    This decorator enforces three guarantees:
    1. The input `data` must be either a numpy.ndarray or a torch.Tensor.
    2. Its number of dimensions must be one of the allowed values.
    3. The wrapped function always receives a numpy array, even if the user
       originally passed a torch tensor.

    After the wrapped function returns, the decorator:
    1. Converts the output back to a torch.Tensor **if and only if** the input
      was originally a tensor.
    2. Restores the original dtype, device, and requires_grad settings.

    Parameters
    ----------
    allowed_dims : int | list[int]
        Allowed number(s) of dimensions for the input data.

    Usage
    -----
    @validate_data([2, 3])
    def my_function(data):
        # data is guaranteed to be a numpy array with ndim in {2, 3}
        return data.mean(axis=0)

    Notes
    -----
    - The decorated function must accept `data` as its first argument.
    - The function must return a single numpy-compatible object.
    - If the passed data has fewer dimensions, it will be automatically
      expanded in the first dimension to match the higher-dimensional
      structure.

    Raises
    ------
    RuntimeError
        If the signature of the decorated function or method is invalid.
    TypeError
        If the input is not a numpy array or torch tensor.
    ValueError
        If the input dimensionality is not allowed.
    """
    if isinstance(allowed_dims, int):
        allowed_dims = [allowed_dims]
    max_dim = max(allowed_dims)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            # Detect if method or function
            if len(args) == 0:
                raise RuntimeError(
                    "validate_data decorator requires at least one argument."
                )

            # If method: args[0] = self, args[1] = data
            # If function: args[0] = data
            if hasattr(args[0], "__class__"):
                if len(args) < 2:
                    raise RuntimeError(
                        "Method must receive data as second argument"
                    )
                data = args[1]
                prefix = args[:1]   # self
                suffix = args[2:]   # remaining args
            else:
                data = args[0]
                prefix = ()
                suffix = args[1:]

            # Track whether input was a torch tensor
            input_was_tensor = torch.is_tensor(data)

            if input_was_tensor:
                original_dtype = data.dtype
                original_device = data.device
                original_requires_grad = data.requires_grad
                data_np = data.detach().cpu().numpy()
            elif isinstance(data, np.ndarray):
                data_np = data
            else:
                raise TypeError(
                    "Input must be numpy.ndarray or torch.Tensor, "
                    f"got {type(data)}"
                )

            # Check dimensions
            if data_np.ndim not in allowed_dims:
                raise ValueError(
                    f"Input has {data_np.ndim} dimensions, "
                    f"but allowed dimensions are {allowed_dims}"
                )

            # Exapand dimensions
            squeeze_axes = None
            if data_np.ndim != max_dim:
                new_shape = (1,) * (max_dim - data.ndim) + data.shape
                data_np = data_np.reshape(new_shape)
                squeeze_axes = range(max_dim - data.ndim)

            # Call the wrapped function with numpy data
            result = func(*prefix, data_np, *suffix, **kwargs)

            # Restore original dimensions
            if squeeze_axes is not None:
                result = result.squeeze(axis=tuple(squeeze_axes))

            # Convert back to torch if needed
            if input_was_tensor:
                result = torch.tensor(
                    result,
                    dtype=original_dtype,
                    device=original_device,
                    requires_grad=original_requires_grad,
                )

            return result

        return wrapper
    return decorator
