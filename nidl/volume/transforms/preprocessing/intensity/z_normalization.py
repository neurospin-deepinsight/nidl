import torch
import numpy as np
# Local import
from .....transform import Transform, TypeTransformInput


class Normalization(Transform):
    def __init__(self, mean=0.0, std=1.0, eps=1e-8, **kwargs):
        """
        Parameters
        ----------
        mean: float
            Expected mean of data.
        
        std: float, default=1.0
            Expected output standard deviation.

        eps: float, default=1e-8
            Small float added to the standard deviation to avoid numerical errors.

        kwargs: dict
            Keyword arguments given to neuroclav.transforms.base.Transform
        """
        self.mean=mean
        self.std=std
        self.eps=eps
        super().__init__(**kwargs)

    def parse_data(self, data: TypeTransformInput) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Unexpected type: %s" % type(data))

    def apply_transform(self, arr: np.ndarray) -> np.ndarray:
        return self.std * (arr - np.mean(arr))/(np.std(arr) + self.eps) + self.mean


class ZNormalization(Normalization):
    def __init__(self, eps=1e-8, **kwargs):
        super().__init__(mean=0.0, std=1.0, eps=eps, **kwargs)
