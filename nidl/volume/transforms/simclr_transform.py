import numpy as np
from typing import Union, Tuple
# Local imports
from ...transform import Transform, Compose
from .preprocessing.intensity.z_normalization import ZNormalization
from .augmentation.spatial.random_flip import RandomFlip
from .augmentation.spatial.random_resized_crop import RandomResizedCrop
from .augmentation.intensity.random_blur import RandomBlur

class SimCLRTransform(Transform):

    """Implements the transformations for SimCLR [1] adapted to 3D medical imaging. 
    In particular, no color jittering or grayscale are applied to these data.
    Additionally, data are z-normalized to have zero mean and unit variance.

    It applies the following transformations:   
      - Random resized crop
      - Random flip (along all axis)
      - Gaussian blur
      - z-normalization

    [1] A Simple Framework for Contrastive Learning of Visual Representations, Chen et al., ICML 2020
    """

    def __init__(self,
                 input_size: Union[int, Tuple[int, ...]]=(1, 128, 128, 128),
                 crop_min_scale: float=0.08,
                 flip_axes: Union[int, Tuple[int, ...]]=(0, 1, 2),
                 flip_prob: float=0.5,
                 blur_std: Union[float, Tuple[float, float]]=(0.1, 2.0),
                 blur_prob: float=0.5,
                 normalize: bool=True):
        """
        Parameters
        ----------
        input_size: int or tuple of int, default=(1, 128, 128, 128)
            Target image shape in voxels.
        
        crop_min_scale: float, default=0.08
            Minimum size of the randomized crop relative to input_size.
        
        flip_axes: int or tuple[int], default=(0,1,2)
            Index or tuple of indices of the spatial dimension along which the image
            might be flipped.
        
        flip_prob: float, default=0.5
            Probability of flipping the image.
        
        blur_std: float or (float, float), default=(0.1, 2.0)
            Range of standard deviations of the Gaussian kernels used to blur the image.
        
        blur_prob: float, default=0.5
            Probability of blurring the image. 

        normalize: bool, default=True
            If True, applies ZNormalization to the output image. 
        """
        super().__init__()

        transform = [
            RandomResizedCrop(size=input_size, scale=(crop_min_scale, 1.0)),
            RandomFlip(axes=flip_axes, p=flip_prob),
            RandomBlur(std=blur_std, p=blur_prob) 
        ]
        if normalize:
            transform.append(ZNormalization())
        self.transform = Compose(transform)
        
    def apply_transform(self, arr: np.ndarray) -> np.ndarray:
        return self.transform(arr)
    
