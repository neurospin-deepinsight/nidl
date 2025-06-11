import random
import numpy as np
import torch
from typing import Union
# Local import
from .....transform import Transform, TypeTransformInput


class RandomFlip(Transform):
    """Reverse the order of elements in an image along the given axis. """

    def __init__(self, 
                 axes: Union[int, tuple[int, ...]] = 0,
                 **kwargs):
        """
        Parameters
        ----------
        axes: index or tuple of indices of the spatial dimension along which the image
            might be flipped. If a tuple of indices is given, only one axis is randomly
            chosen to perform the flip around this axis.

        kwargs: dict
            See neuroclav.data.transforms.base.Transform for additional keyword arguments
        """
        super().__init__(**kwargs)
        self.axes = self._parse_axes(axes)
    
    def _parse_axes(self, axes: Union[int, tuple[int, ...]]):
        axes_tuple = tuple(axes)
        for axis in axes_tuple:
            is_int = isinstance(axis, int)
            valid_number = is_int and axis in (0, 1, 2)
            if not valid_number:
                message = (
                    f'All axes must be 0, 1 or 2, but found "{axis}" with type {type(axis)}'
                )
                raise ValueError(message)
        return axes_tuple
    
    def parse_data(self, data: TypeTransformInput) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Unexpected type: %s"%type(data))

    def apply_transform(self, arr: np.ndarray):
        """
        Parameters
        ----------
        arr: array of shape (W, H, D) or (*, W, H, D)
        """
        possible_axes = np.array(self.axes, int)
        if arr.ndim == 4:
            possible_axes += 1
        selected_axis = random.choice(possible_axes)
        return np.flip(arr, axis=(selected_axis,))