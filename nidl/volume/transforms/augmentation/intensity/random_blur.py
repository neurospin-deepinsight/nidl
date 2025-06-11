import torch
import numpy as np
import random
from typing import Union, Tuple
from scipy.ndimage import gaussian_filter
# Local import
from .....transform import Transform, TypeTransformInput


class RandomBlur(Transform):
    """Blur a 3D image using a random-sized Gaussian filter."""
    def __init__(self, std: Union[float, Tuple[float, ...]], **kwargs):
        """
        Parameters
        ----------
        std: float or tuple[float, ...] 
            Ranges of the standard deviations (s1, s2, s3) of the Gaussian kernels used to 
            blur the image along each axis, where si ~ U(ai, bi) (i=1,2,3). 
            If a single value s is provided, ai=0, bi=s for i=1,2,3
            If two values (a,b) are provided, ai=a, bi=b for i=1,2,3
            If six values (a1, b1, a2, b2, a3, b3) are provided, everything is user-defined.
        kwargs: dict
            Keyword arguments given to neuroclav.transforms.base.Transform
        """
        super().__init__(**kwargs)

        if isinstance(std, list):
            std = tuple(std)
        if isinstance(std, float):
            self.params = (0, std, 0, std, 0, std)
        elif isinstance(std, tuple):
            if len(std) == 2:
                self.params = (std[0], std[1], std[0], std[1], std[0], std[1])
            elif len(std) == 6:
                self.params = std
            else:
                raise ValueError(f"std must have length 2 or 6, got {len(std)}")
        else:
            raise ValueError(f"std must be either a float or tuple of floats, got {type(std)}")

    def sample_uniform_params(self):
        results = []
        for (a, b) in zip(self.params[::2], self.params[1::2]):
            results.append(random.uniform(a, b))
        return tuple(results)
    
    
    def parse_data(self, data: TypeTransformInput) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Unexpected type: %s" % type(data))
    
    def apply_transform(self, arr: np.ndarray):
        stds = self.sample_uniform_params()
        if arr.ndim == 4: # (c, h, w, d)
            return np.stack([gaussian_filter(arr_channel, stds) for arr_channel in arr])
        elif arr.ndim == 3:
            return gaussian_filter(arr, stds)
        else:
            raise ValueError(f"input array must be 3D or 4D, got {arr.ndim}D")

