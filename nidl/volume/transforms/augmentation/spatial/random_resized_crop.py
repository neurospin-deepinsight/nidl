import random
import numpy as np
from typing import Sequence, Tuple, Union
import torch
# Local import
from .....transform import Transform, TypeTransformInput
from ...preprocessing.spatial.resample import Resample
from ...preprocessing.spatial.crop_or_pad import CropOrPad

class RandomResizedCrop(Transform):
    """Crop a random portion of a given 4D (channel dim + 3D spatial dim) array and resize it
    to a given size. It does not modify the field of view.
    It is a generalization of `torchvision.transforms.RandomResizedCrop`
    to 3D case. It also preserves the input aspect ratio.
    """
    def __init__(self, size: Union[int, Sequence[int]],
                 scale: Tuple[float, float]=(0.08, 1.0),
                 **kwargs):
        """
        Parameters
        ----------
        size: int or Sequence of int (W, H, D) or (*, W, H, D) where (*) is channel dimension.
            Expected output shape. If a single integer is given, it applies the same size across
            all dimensions.

        scale: Tuple of float
            Specifies lower and upper bounds for the random area of the crop, before resizing.
            The scale is defined with respect to the area of the original image.

        kwargs: dict
            Keywords arguments passed to Transform.
        """
        super().__init__(**kwargs)
        if isinstance(size, int):
            size = 3 * (size,)
        elif isinstance(size, Sequence):
            if len(size) == 4:
                size = size[1:]
            elif len(size) < 3 or len(size) > 4:
                raise ValueError("Size must be 3D or 4D, got %iD"%len(size))
        else:
            raise ValueError("Unexpected size, got {}".format(size))
        self.size = size
        self.scale = scale

    def parse_data(self, data: TypeTransformInput) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Unexpected type: %s"%type(data))

    def apply_transform(self, arr: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Parameters
        ----------
        arr: array of shape (W, H, D)  or (*, W, H, D)
        
        Returns
        ----------
        array
            Random crop of input array, resized to input shape (field of view is constant).
        """
        #from skimage.transform import resize
        volume = np.prod(arr.shape)
        target_volume = random.uniform(*self.scale) * volume
        target_ratio = (target_volume / volume) ** (1./arr.ndim)
        indexes = []
        for dim in range(arr.ndim):
            crop_size = max(int(round(target_ratio * arr.shape[dim])), 1)
            i = random.randint(0, arr.shape[dim] - crop_size)
            indexes.append(slice(i, i + crop_size))
        # randomly crop image (keep field of view)
        img = arr[tuple(indexes)]
        # resize the image to the input shape
        shape_in = np.asarray(img.shape)
        if len(shape_in) == 4:
            shape_in = shape_in[1:]
        shape_out = np.asarray(self.size)
        resample = Resample(tuple(shape_in/shape_out), interpolation="linear")
        resampled = resample(img)
        resampled_shape = resampled.shape
        if len(resampled_shape) == 4:
            resampled_shape = resampled_shape[1:]
        # Sometimes, the output shape is one voxel too large
        # Probably because Resample uses np.ceil to compute the shape
        if not tuple(resampled_shape) == tuple(shape_out):
            print(f"Warning: crop or pad applied (got {resampled.shape}, expected {shape_out})", flush=True)
            crop_pad = CropOrPad((resampled.shape[0],) + tuple(shape_out))
            return crop_pad(resampled)
        return resampled
    