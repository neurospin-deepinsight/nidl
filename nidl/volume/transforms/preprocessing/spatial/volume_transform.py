from typing import Union, Tuple
import torch
import numpy as np
# Local import
from .....transform import Transform, Compose, TypeTransformInput
from ..spatial.crop_or_pad import CropOrPad
from ..intensity.z_normalization import ZNormalization


class VolumeTransform(Transform):
    """Implements the basic transformations for 3D volume including resizing to a given 
     shape with crop/pad and z-normalization.
    """
    
    def __init__(self,                 
                 input_size: Union[int, Tuple[int, ...]]=(1, 128, 128, 128),
                 normalize: bool=True):
        """
        Parameters
        ----------
        input_size: int or tuple of int, default=(1, 128, 128, 128)
            Target image shape in voxels.
        
        normalize: bool, default=True
            If True, applies ZNormalization to the output image.
        """
        super().__init__()
        transform = [CropOrPad(input_size)]
        if normalize:
            transform.append(ZNormalization())
        self.transform = Compose(transform)
    

    def parse_data(self, data: TypeTransformInput) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Unexpected type: %s"%type(data))

    def apply_transform(self, arr: np.ndarray) -> np.ndarray:
        return self.transform(arr)
