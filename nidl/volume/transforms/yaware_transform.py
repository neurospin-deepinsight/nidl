import numpy as np
from typing import Union, Tuple
# Local imports
from ...transform import Transform, Compose
from .preprocessing.spatial.crop_or_pad import CropOrPad
from .augmentation.spatial.random_cutout import RandomCutout
from .augmentation.spatial.random_flip import RandomFlip
from .augmentation.spatial.random_resized_crop import RandomResizedCrop
from .augmentation.intensity.random_blur import RandomBlur
from .augmentation.intensity.random_noise import RandomNoise
from .preprocessing.intensity.z_normalization import ZNormalization


class yAwareTransformLight(Transform):
    """Implements the transformations for y-Aware Contrastive Learning [1] (light version).
    It applies only random cutout.

    Update (2025): multiple areas are removed from the input image.

    [1] Contrastive Learning with Continuous Proxy Meta-Data for 3D MRI Classification, Dufumier et al., MICCAI 2021
    """

    def __init__(self,
                 input_size: Union[int, Tuple[int]]=(1, 128, 128, 128),
                 min_scale: float=0.02,
                 max_scale: float=0.33,
                 num_iterations: int=10,
                 value: float=0.0,
                 normalize: bool=True,
                 p: float=0.5):
        """
        Parameters
        ----------
        input_size: int or tuple of int, default=(1, 128, 128, 128)
            Target image shape in voxels.

        min_scale: float, default=0.02
            Minimum size of the random erased area relative to the input volume.

        max_scale: float, default=0.33
            Maximum size of the random erased area relative to the input volume.
        
        num_iterations: int, default=10
            Number of erased areas.
        
        value: float, default=0.0
            Value used to replace the erased area.

        normalize: bool, default=True
            If True, applies Z-normalization to the input image.

        p: float between 0 and 1, default=0.5
            Probability to apply the cutout.
        """
        super().__init__()

        transform = [
            CropOrPad(input_size),
            RandomCutout((min_scale, max_scale), num_iterations, value, p=p)
            ]
        if normalize:
            transform.append(ZNormalization())
        self.transform = Compose(transform)
        
    def apply_transform(self, arr: np.ndarray) -> np.ndarray:
        return self.transform(arr)


class yAwareTransformStrong(Transform):

    """Implements the transformations for y-Aware Contrastive Learning [1] (strong version). 
    It applies the following transformations:   
      - Random resized crop  
      - Random Flip
      - Gaussian blur
      - Gaussian noise
      - Random cutout
    
    Update (2025): multiple areas are removed from the input image.

    [1] Contrastive Learning with Continuous Proxy Meta-Data for 3D MRI Classification, Dufumier et al., MICCAI 2021
    """

    def __init__(self,
                 input_size: Union[int, Tuple[int, ...]]=(1, 128, 128, 128),
                 crop_min_scale: float=0.08,
                 flip_axes: Union[int, Tuple[int, ...]]=(0, 1, 2),
                 flip_prob: float=0.5,
                 blur_std: Union[float, Tuple[float, float]]=(0.1, 1.0),
                 blur_prob: float=0.5,
                 noise_std: Union[float, Tuple[float, float]]=(0.1, 1.0),
                 noise_prob: float=0.5,
                 min_scale: float=0.02,
                 max_scale: float=0.33,
                 num_iterations: int=10,
                 cutout_value: float=0.0,
                 cutout_prob: float=0.5,
                 normalize: bool=True
                 ):
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
        
        blur_std: float or (float, float), default=(0.1, 1.0)
            Range of standard deviations of the Gaussian kernels used to blur the image.
        
        blur_prob: float, default=0.5
            Probability of blurring the image. 
        
        noise_std: float or (float, float), default=(0.1, 1.0)
            Range of standard deviations of the Gaussian noise added to the image.
        
        noise_prob: float, default=0.5
            Probability of adding noise.

        min_scale: float, default=0.02
            Minimum size of the random erased area relative to the input volume.

        max_scale: float, default=0.33
            Maximum size of the random erased area relative to the input volume.
        
        num_iterations: int, default=10
            Number of erased areas.
        
        cutout_value: float, default=0.0
            Value used to replace the erased area.
        
        cutout_prob: float, default=0.5
            Probability of applying cutout.

        normalize: bool, default=True
            If True, applies ZNormalization to the output image. 
        """
        super().__init__()

        transform = [
            RandomResizedCrop(size=input_size, scale=(crop_min_scale, 1.0)),
            RandomFlip(axes=flip_axes, p=flip_prob),
            RandomBlur(std=blur_std, p=blur_prob),
            RandomNoise(std=noise_std, p=noise_prob),
            RandomCutout((min_scale, max_scale), num_iterations, cutout_value, p=cutout_prob)   
        ]
        if normalize:
            transform.append(ZNormalization())
        self.transform = Compose(transform)
        
    def apply_transform(self, arr: np.ndarray) -> np.ndarray:
        return self.transform(arr)
    





