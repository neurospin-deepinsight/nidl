##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import numbers
from typing import Union

import numpy as np
import torch

from .....transforms import TypeTransformInput, VolumeTransform


class RandomErasing(VolumeTransform):
    """Randomly erases boxes in a 3d volume.

    Randomly selects one or multiple boxes in input data and erases their
    values (i.e. random erasing [1]_, very similar to cutout [2]_ but with
    arbitrary aspect ratio). It is an extension of
    `torchvision.transforms.RandomErasing` to the 3d case and it can
    eventually erase multiple random boxes. It handles `np.ndarray` or
    `torch.Tensor` as input and returns a consistent output (same type
    and shape).

    Parameters
    ----------
    scale: tuple of (float, float), default=(0.02, 0.33)
        Range of proportion of erased area against input data.
    ratio: tuple of (float, float), default=(1.0, 3.0)
        Range of aspect ratio of erased area (min, max).
    num_iterations: int, default=1
        Number of erased areas.
    value: float, "mean" or "random", default=0.0
        Erasing value. If "random", erases each voxel with random values
        normally distributed. If "mean", replaces each voxel with the mean
        value of the erased area, preserving the global statistics.
    inplace: bool, default=False
        If true, makes the transformation inplace, i.e. it modifies
        the input data directly.
    kwargs: dict
        Additional keyword.

    Notes
    -----
    In 3d, we define the "aspect ratio" as the ratio between each dimension
    size relatively to their geometric mean. It is a simple generalization
    from 2d to nd and we don't particularize any dimension. The aspect ratio
    is sampled three times for a 3d volume (one for each dimension).

    References
    ----------
    .. [1] Zhong, Z., Zheng, L., Kang, G., Li, S., & Yang, Y. (2020).
           Random Erasing Data Augmentation.
           In AAAI Conference on Artificial Intelligence.
           https://arxiv.org/abs/1708.04896

    .. [2] DeVries, T., & Taylor, G. W. (2017).
           Improved Regularization of Convolutional Neural Networks with Cutout
           https://arxiv.org/abs/1708.04552
    """

    def __init__(
        self,
        scale: tuple[float, float] = (0.02, 0.33),
        ratio: tuple[float, float] = (1.0, 3.0),
        num_iterations: int = 1,
        value: Union[float, str] = 0.0,
        inplace: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale = self._parse_range(scale, check_min=0, check_max=1)
        self.ratio = self._parse_range(ratio, check_min=0)
        self.num_iterations = num_iterations
        self.value = value
        self.inplace = inplace

    @staticmethod
    def _sample_3d_box(in_shape, scale, ratio) -> list[slice]:
        """Randomly sample a 3d box to erase from input."""

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
        if box is None:
            # If not possible, returns empty box.
            return [slice(0, 0) for _ in in_shape]
        return box

    def apply_transform(self, data: TypeTransformInput) -> TypeTransformInput:
        """Randomly erase boxes in the data.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            The input data with shape :math:`(C, H, W, D)` or
            :math:`(H, W, D)`. Erased boxes are equal the same across channels.

        Returns
        -------
        data: np.ndarray or torch.Tensor
            Data with erased boxes.
        """
        if not self.inplace:
            if isinstance(data, torch.Tensor):
                data = data.clone()
            else:
                data = np.copy(data)

        for _ in range(self.num_iterations):
            in_shape = data.shape
            if data.ndim == 4:  # remove channel
                in_shape = in_shape[1:]
            box_to_erase = RandomErasing._sample_3d_box(
                in_shape, self.scale, self.ratio
            )
            if data.ndim == 4:
                box_to_erase.insert(0, slice(None))

            slicer = tuple(box_to_erase)
            if isinstance(self.value, numbers.Number):
                data[slicer] = self.value
            elif self.value == "mean":
                data[slicer] = data[slicer].mean()
            elif self.value == "random":
                region = data[slicer]
                if isinstance(data, torch.Tensor):
                    data[slicer] = torch.randn_like(region)
                else:
                    data[slicer] = np.random.randn(*region.shape)
            else:
                raise ValueError(
                    f"`value` must be scalar (int or float), 'mean' or"
                    f" 'random', got {self.value}"
                )
        return data
