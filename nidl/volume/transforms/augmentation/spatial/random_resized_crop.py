##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from typing import Union

import numpy as np

from .....transforms import TypeTransformInput, VolumeTransform
from ...preprocessing.spatial.resize import Resize


class RandomResizedCrop(VolumeTransform):
    """Crop a random portion of a 3d volume and resize it.

    It is a generalization of `torchvision.transforms.RandomResizedCrop`
    to the 3d case.

    It handles `np.ndarray` or `torch.Tensor` as input and
    returns a consistent output (same type).
    
    Parameters
    ----------
    target_shape: int or tuple of (int, int, int)
        Expected output shape. If int, apply the same size across all
        dimensions.
    scale: tuple of (float, float), default=(0.08, 1.0)
        Specifies lower and upper bounds for the random area of the crop,
        before resizing. The scale is defined with respect to the area of the
        original image.
    ratio: tuple of (float, float), default=(1.0, 1.33)
        Range of the aspect ratio of the crop, before resizing.
    interpolation: str in {'nearest', 'linear', 'bspline', 'cubic', \
        'gaussian', 'label_gaussian', 'hamming', 'cosine', 'welch', \
        'lanczos', 'blackman'}, default='linear'
        Interpolation techniques available in ITK. `linear`, the default in
        nidl for scalar images, offers a good compromise between image
        quality and speed and is a solid choice for data augmentation during
        training.
    kwargs: dict
        Keyword arguments given.
    
    Notes
    -----
    In 3d, we define the "aspect ratio" as the ratio between each dimension
    size relatively to their geometric mean. It is a simple generalization
    from 2d to nd and we don't particularize any dimension. The aspect ratio
    is sampled three times for a 3d volume (one for each dimension).

    """

    def __init__(
        self,
        target_shape: Union[int, tuple[int, int, int]],
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (0.75, 1.33),
        interpolation: str = "linear",
        **kwargs,
    ):
        """ """
        super().__init__(**kwargs)

        self.target_shape = self._parse_shape(target_shape, length=3)
        self.scale = self._parse_range(scale, check_min=0, check_max=1)
        self.ratio = self._parse_range(ratio, check_min=0)
        self.interpolation = interpolation

    @staticmethod
    def _sample_3d_box(in_shape, scale, ratio) -> list[slice]:
        """Randomly sample a 3d box to crop from input."""

        def try_sample_box(ratio):
            volume = np.prod(in_shape)
            # Sample a target volume
            target_volume = np.random.uniform(*scale) * volume
            # Sample one aspect ratio per dimension
            log_ratio = np.log(np.array(ratio))
            sampled_ar = np.exp(np.random.uniform(*log_ratio, size=3))
            # Normalize aspect ratios to keep geometric mean = 1
            sampled_ar /= np.cbrt(np.prod(sampled_ar))

            box = []  # list of slices
            cbrt_volume = np.cbrt(target_volume)
            for ar, size in zip(sampled_ar, in_shape):
                box_size = round(cbrt_volume * ar)
                if box_size > size:
                    return None
                i = np.random.randint(0, size - box_size + 1)
                box.append(slice(i, i + box_size))
            return box

        # Sample boxes until it fits into the volume.
        for _ in range(10):
            box = try_sample_box(ratio)
            if box is not None:
                return box
        # If not possible, fallback to unit ratio.
        box = try_sample_box((1.0, 1.0))
        if box is None:  # returns whole volume
            return [slice(None) for _ in range(in_shape)]
        return box

    def apply_transform(self, data: TypeTransformInput) -> TypeTransformInput:
        """Crop and resize the input volume.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            The input data with shape :math:`(C, H, W, D)` or
            :math:`(H, W, D)`. Cropped area is the same across channels.

        Returns
        -------
        data: np.ndarray or torch.Tensor
            Cropped and resized data. Output type is the same as input.
        """
        in_shape = data.shape
        if data.ndim == 4:  # remove channel
            in_shape = in_shape[1:]
        box = RandomResizedCrop._sample_3d_box(
            in_shape, self.scale, self.ratio
        )
        if data.ndim == 4:  # re-insert channel
            box.insert(0, slice(None))

        # randomly crop the volume
        data = data[tuple(box)]

        # resample the volume to match target shape
        resample = Resize(
            self.target_shape,
            interpolation=self.interpolation,
        )
        resampled = resample(data)
        return resampled
