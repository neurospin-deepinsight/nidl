import random
import numpy as np
from typing import Tuple
import torch
# Local import
from .....transform import Transform, TypeTransformInput


class RandomCutout(Transform):
    """
        Randomly selects one or multiple rectangle regions in input image and erases its pixels/voxels.
        It is an extension of `torchvision.transforms.RandomErasing` to the multi-dimensional case with
        eventually multiple random rectangles to erase. The original input ratio is preserved for each
        erased rectangles.
        Papers:
        Improved Regularization of Convolutional Neural Networks with Cutout, DeVries, 2017
        Random Erasing Data Augmentation, Zhong et al., AAAI 2020
    """

    def __init__(self, 
                 scale: Tuple[float, float]=(0.02, 0.33), 
                 num_iterations: int=1,
                 value: float=0.0, 
                 inplace: bool=False, 
                 **kwargs):
        """
        Parameters
        ----------
        scale: Tuple of (float, float)
            Range of proportion of erased area against input image
        
        num_iterations: int
            Number of erased areas
        
        value: float
            Erasing value.
        
        inplace: boolean
            If true, makes the transformation inplace
        
        kwargs: dict
            Keyword args given to super()
        """
        super().__init__(**kwargs)
        self.scale = scale
        self.num_iterations = num_iterations
        self.value = value
        self.inplace = inplace

    def parse_data(self, data: TypeTransformInput) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Unexpected type: %s"%type(data))

    def _erase_area(self, arr: np.ndarray) -> np.ndarray:
        arr_shape = np.array(arr.shape)
        volume = np.prod(arr_shape)
        occluded_volume = random.uniform(*self.scale) * volume
        ratio = (occluded_volume / volume) ** (1./arr.ndim)
        indexes = []
        for dim in range(arr.ndim):
            cutout_size = max(int(round(ratio * arr_shape[dim])), 1)
            i = random.randint(0, arr_shape[dim] - cutout_size)
            indexes.append(slice(i, i + cutout_size))
        arr[tuple(indexes)] = self.value
        return arr

    def apply_transform(self, arr: np.ndarray) -> np.ndarray:
        if not self.inplace:
            arr = np.copy(arr)
        for _ in range(self.num_iterations):
            arr = self._erase_area(arr)
        return arr
    
