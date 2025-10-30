##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from typing import Union

import numpy as np
import SimpleITK as Stk
import torch

from .resample import Resample, TypeTransformInput


class Resize(Resample):
    """Resize a 3d volume to match a target shape.

    This transformation resizes a 3d volume to a new target shape,
    implicitely modifying the physical spacing. Internally, it uses SimpleITK
    for fast and robust resampling. It handles :class:`numpy.ndarray` or
    :class:`torch.Tensor` as input and returns a consistent output (same type).
    Input shape must be :math:`(C, H, W, D)` or :math:`(H, W, D)`.
    
    Parameters
    ----------
    target_shape: int or tuple of (int, int, int)
        Output shape :math:`(H', W', D')`. If int is given, it sets
        :math:`H'=W'=D'`.

    interpolation: str in {'nearest', 'linear', 'bspline', 'cubic', \
        'gaussian', 'label_gaussian', 'hamming', 'cosine', 'welch', \
        'lanczos', 'blackman'}, default='linear'

        Interpolation techniques available in ITK. `linear`, the default in
        nidl for scalar images, offers a good compromise between image
        quality and speed and is a solid choice for data augmentation during
        training.
        Methods such as `bspline` or `lanczos` produce high-quality results
        but are slower and best used during offline preprocessing. `nearest`
        is very fast but gives poorer results for scalar images; however, it is
        the default for label maps, as it preserves categorical values.
        For a full comparison of interpolation methods, see [1]_.
        Descriptions of available methods:

            - `nearest`: Nearest-neighbor interpolation.
            - `linear`: Linear interpolation.
            - `bspline`: B-spline of order 3 (cubic).
            - `cubic`: Alias for `bspline`.
            - `gaussian`: Gaussian interpolation :math:`\\sigma=0.8,\\alpha=4`.
            - `label_gaussian`: Gaussian interpolation for label maps
              (:math:`\\sigma=1, \\alpha=1`).
            - `hamming`: Hamming-windowed sinc kernel.
            - `cosine`: Cosine-windowed sinc kernel.
            - `welch`: Welch-windowed sinc kernel.
            - `lanczos`: Lanczos-windowed sinc kernel.
            - `blackman`: Blackman-windowed sinc kernel.

    **kwargs : dict
        Keyword arguments given to :class:`nidl.transforms.Transform`.

    References
    ----------
    .. [1] Meijering et al. (1999), "Quantitative Comparison of
           Sinc-Approximating Kernels for Medical Image Interpolation."
    """

    def __init__(
        self,
        target_shape: Union[int, tuple[int, int, int]],
        interpolation: str = "linear",
        **kwargs,
    ):
        super().__init__(interpolation=interpolation, **kwargs)
        self.target_shape = self._parse_shape(target_shape, length=3)

    def apply_transform(self, data: TypeTransformInput) -> TypeTransformInput:
        """Resize the input volume.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            The input data with shape :math:`(C, H, W, D)` or
            :math:`(H, W, D)`. The channel dimension is never resized.

        Returns
        ----------
        data: np.ndarray or torch.Tensor
            Resampled data with shape :math:`(H', W', D')`  or
            :math:`(C, H', W', D')` and same type as input.

        """

        in_shape = data.shape

        if len(in_shape) == 4:
            in_shape = in_shape[1:]

        is_data_tensor = False
        if isinstance(data, torch.Tensor):  # Computations are performed on CPU
            is_data_tensor = True
            dtype, device = data.dtype, data.device
            data = data.detach().cpu().numpy()

        floating_sitk = Resample.as_sitk(data, np.eye(4))

        resampler = Stk.ResampleImageFilter()
        resampler.SetInterpolator(self.interpolator)
        reference_image = Resize.get_reference_image(
            floating_sitk,
            self.target_shape,
        )
        resampler.SetReferenceImage(reference_image)
        resampled = resampler.Execute(floating_sitk)
        resampled = Resample.from_sitk(resampled, dim=data.ndim)

        if is_data_tensor:
            resampled = torch.as_tensor(resampled, dtype=dtype, device=device)
        return resampled

    @staticmethod
    def get_reference_image(
        floating_sitk: Stk.Image,
        new_shape: tuple[int, int, int],
    ) -> Stk.Image:
        old_spacing = np.array(floating_sitk.GetSpacing())
        old_shape = np.array(floating_sitk.GetSize())
        new_shape = np.array(new_shape)
        new_spacing = old_shape / new_shape * old_spacing
        new_shape = np.array(new_shape).astype(np.uint16)
        new_origin_index = 0.5 * (new_spacing / old_spacing - 1)
        new_origin_lps = floating_sitk.TransformContinuousIndexToPhysicalPoint(
            new_origin_index,
        )
        reference = Stk.Image(
            new_shape.tolist(),
            floating_sitk.GetPixelID(),
            floating_sitk.GetNumberOfComponentsPerPixel(),
        )
        reference.SetDirection(floating_sitk.GetDirection())
        reference.SetSpacing(new_spacing.tolist())
        reference.SetOrigin(new_origin_lps)
        return reference
