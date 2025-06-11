import numpy as np
from typing import Sequence, Union
import torch
# Local import
from .....transform import Transform, TypeTransformInput


class CropOrPad(Transform):
    """
    Modify the field of view by cropping and/or padding to match the target shape
    """
    def __init__(self, size: Union[int, Sequence[int]],
                 padding_mode: str='constant',
                 constant_values: Union[Sequence, float]=0.0,
                 **kwargs):
        """
        Parameters
        ----------
        size: int or Sequence of int
            Expected output shape
        
        padding_mode: str
            See possibe modes in Numpy doc `np.pad`. Should be in
            {'edge', 'maximum', 'constant', 'mean', 'median',
            'minimum', 'reflect', 'symmetric', 'wraps'}
        
        constant_values: Sequence of float
            See possible values in Numpy doc. Constant values applied if 'constant' mode is set
        
        kwargs: dict
            parameters given to Transform
        """
        super().__init__(**kwargs)

        self.size = size
        self.padding_mode = padding_mode
        self.constant_values = constant_values

    def parse_data(self, data: TypeTransformInput) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Unexpected type: %s"%type(data))

    def apply_transform(self, arr: np.ndarray) -> np.ndarray:
        crop_bounding_box = []
        pad_widths = []
        arr_shape = arr.shape
        size = self.size
        if isinstance(self.size, int):
            size = arr.ndim * (self.size,)
        if len(size) != arr.ndim:
            raise ValueError("size should be of the same length as number of " \
            "dimensions of the input array (%d), got %d"%(arr.ndim, len(size)))
        for dim in range(arr.ndim):
            if size[dim] >= arr_shape[dim]:
                crop_bounding_box.append(slice(0, arr_shape[dim]))
                pad_before = (size[dim] - arr_shape[dim]) // 2
                pad_after = max(size[dim] - (arr_shape[dim] + pad_before), 0)
                pad_widths.append((pad_before, pad_after))
            else:
                crop_from = (arr_shape[dim] - size[dim]) // 2
                crop_until = crop_from + size[dim]
                crop_bounding_box.append(slice(crop_from, crop_until))
                pad_widths.append((0, 0))
        # First crop the image
        arr = arr[tuple(crop_bounding_box)]
        # Then pad
        if self.padding_mode == "constant":
            arr = np.pad(arr, pad_widths, mode=self.padding_mode, constant_values=self.constant_values)
        else:
            arr = np.pad(arr, pad_widths, mode=self.padding_mode)
        return arr
