from __future__ import annotations

from typing import Union

from ....transforms import VolumeTransform
from .crop_or_pad import CropOrPad
from .resize import Resize


class Letterbox(VolumeTransform):
    """
    Implementation of the letterbox transform introduced in YOLO [1].
    It resizes the input volume to an isotropic target shape
    while preserving the aspect ratio. This is done to minimize the
    distortion of volume's objects (e.g. anatomical structures) induced
    by the resizing process.

    The volume is first resized to an intermediate shape with a scale
    factor such that the longest input dimension is smaller than the
    target one. The resulting volume is then padded to the target shape.

    Parameters
    -----------
    target_shape: int or tuple of (int, int, int)
        Output shape :math:`(H', W', D')`. If int is given, it sets
        :math:`H'=W'=D'`.
        Target shape must be isotropic.
        
    padding_mode: str in {'edge', 'maximum', 'constant', 'mean', 'median',\
        'minimum', 'reflect', 'symmetric'}
        Possible modes for padding. See more infos in the
        `Numpy documentation
        <https://numpy.org/doc/stable/reference/generated/numpy.pad.html>`_.
            
    constant_values: float or tuple[float, float, float]
        The values to set the padded values for each
        axis if the padding mode is 'constant'.
        
    interpolation: str in {'nearest', 'linear', 'bspline', 'cubic', \
        'gaussian', 'label_gaussian', 'hamming', 'cosine', 'welch', \
        'lanczos', 'blackman'}, default='linear'

        Interpolation techniques available in ITK. See the documentation of
        Nidl's resize transform for more details.

    References
    ----------
    [1] Redmon, J., et al., "You Only Look Once: ..." CVPR, 2016.
    https://arxiv.org/abs/1506.02640
    """
    def __init__(
        self,
        target_shape: Union[int, tuple[int, int, int]],
        padding_mode: str = "constant",
        constant_values: Union[float, tuple[float, float, float]] = 0.0,
        interpolation: str = "linear"
    ):
        super().__init__()

        self.target_shape = self._parse_shape(target_shape, length=3)
        if (self.target_shape[0] == self.target_shape[1]
                and self.target_shape[1] == self.target_shape[2]):
            self.interpolation = interpolation
            self.padding_transform = CropOrPad(
                self.target_shape, padding_mode, constant_values)
        else:
            raise ValueError("Target shape must be isotropic\n")
    
    def compute_intermediate_shape(self, original_shape):
        longest_dim = max(original_shape)
        # The same scale factor is applied across all dimensions
        # to preserve aspect ratio
        scale_factor = self.target_shape[0] / longest_dim
        resized_shape = tuple(
            round(dim * scale_factor) for dim in original_shape)

        return resized_shape
    
    def apply_transform(self, data_parsed, *args, **kwargs):
        in_shape = data_parsed.shape[-3:]

        # 1. Compute scale factor (same for all dimensions to preserve ratio)
        resized_shape = self.compute_intermediate_shape(in_shape)

        # 2. Resize image
        resize_transform = Resize(resized_shape, self.interpolation)
        resized_image = resize_transform(data_parsed)

        # 3. Padding to target shape
        return self.padding_transform(resized_image)
