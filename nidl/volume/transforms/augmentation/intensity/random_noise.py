import torch
import numpy as np
import random
from typing import Union, Tuple
# Local import
from .....transform import Transform, TypeTransformInput


class RandomNoise(Transform):
    """Add Gaussian noise to input data with random parameters."""

    def __init__(self, 
                 mean: float=0.0, 
                 std: Union[float, Tuple[float, float]]=(0.1, 1.0),
                 **kwargs):
        """
        Parameters
        ----------
        mean: float
            Mean of the Gaussian distribution from which the noise is sampled. 

        std: float or (float, float)
            Range of the standard deviation (a, b) of the Gaussian distribution 
            from which the noise is sampled sigma ~ U(a, b). 
            If a single value b is provided, a=0, b=b 
            If two values (a,b) are provided, a=a, b=b

        kwargs: dict
            Keyword arguments given to neuroclav.transforms.base.Transform
        """
        super().__init__(**kwargs)

        self.mean = mean

        if isinstance(std, list):
            std = tuple(std)
        if isinstance(std, float):
            self.std_range = (0, std)
        elif isinstance(std, tuple):
            if len(std) == 2:
                self.std_range = std
            else:
                raise ValueError(f"std must have length 2, got {len(std)}")
        else:
            raise ValueError(f"std must be either a float or tuple of floats, got {type(std)}")
    
    def parse_data(self, data: TypeTransformInput) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Unexpected type: %s" % type(data))
    
    def apply_transform(self, arr: np.ndarray):
        std = random.uniform(self.std_range[0], self.std_range[1])
        return np.random.normal(self.mean, std, size=arr.shape) + arr 
