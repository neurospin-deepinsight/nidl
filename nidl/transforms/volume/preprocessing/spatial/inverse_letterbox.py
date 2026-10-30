from __future__ import annotations

from typing import Union

from ....transforms import VolumeTransform
from .crop_or_pad import CropOrPad
from .resize import Resize


class InverseLetterbox(VolumeTransform):
    """
    Inverse of the letterbox transform introduced in YOLO [1].
    The input volume is first cropped back to an intermediate shape
    and then resized to the original shape.

    This transform might be useful for tasks that need to be solved
    in the original volumes space (e.g. segmentation).

    Parameters
    -----------
    standard_shape: int or tuple of (int, int, int)
        Shape of the input volume :math:`(H', W', D')`.
        If int is given, it sets :math:`H'=W'=D'`.
        Target shape must be isotropic.
        
    interpolation: str in {'nearest', 'linear', 'bspline', 'cubic', \
        'gaussian', 'label_gaussian', 'hamming', 'cosine', 'welch', \
        'lanczos', 'blackman'}, default='linear'

        Interpolation techniques available in ITK. See the documentation of
        Nidl's resize transform for more details.


    References
    ----------
    [1] Redmon, J., et al., "You Only Look Once: ..." CVPR, 2016. https://arxiv.org/abs/1506.02640
    """
    def __init__(
        self,
        standard_shape: Union[int, tuple[int, int, int]],
        interpolation: str = "linear"
    ):
        super().__init__()

        self.standard_shape = self._parse_shape(standard_shape, length=3)
        if (self.standard_shape[0] == self.standard_shape[1]
                and self.standard_shape[1] == self.standard_shape[2]):
            self.interpolation = interpolation
        else:
            raise ValueError("standard_shape must be isotropic\n")
    
    def compute_intermediate_shape(self, original_shape):
        longest_dim = max(original_shape)
        scale_factor = self.standard_shape[0] / longest_dim
        resized_shape = tuple(
            round(dim * scale_factor) for dim in original_shape)

        return resized_shape
    
    def apply_transform(self, data_parsed, original_shape, *args, **kwargs):
        
        # 1. Compute post-scaling shape
        resized_shape = self.compute_intermediate_shape(original_shape)

        # 2. Crop back from standard shape to rescaled shape
        cropping_transform = CropOrPad(resized_shape)
        cropped_image = cropping_transform(data_parsed)

        # 3. Re-size to original shape
        resize_transform = Resize(original_shape, self.interpolation)
        return resize_transform(cropped_image)
