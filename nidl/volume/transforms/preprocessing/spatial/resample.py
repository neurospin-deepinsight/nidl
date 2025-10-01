##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from collections.abc import Iterable
from numbers import Number
from typing import Optional, Union

import numpy as np
import SimpleITK as Stk
import torch
from nibabel.orientations import aff2axcodes
from sklearn.utils.validation import check_array

from .....transforms import TypeTransformInput, VolumeTransform


class Resample(VolumeTransform):
    """Resample a 3d volume to a different physical space.

    This transformation resamples a 3d (or 4d with channels) volume to
    a new spacing, effectively changing its shape. It uses a provided
    RAS-affine matrix to interpret voxel coordinates in physical space.
    Check `Nibabel documentation on image orientation
    <https://nipy.org/nibabel/coordinate_systems.html>`_.

    Internally, it uses SimpleITK for fast and robust resampling.

    It handles :class:`np.ndarray` or :class:`torch.Tensor` as input and
    returns a consistent output (same type).

    Input shape must be :math:`(C, H, W, D)` or :math:`(H, W, D)`.

    Parameters
    ----------
    target: float or tuple of floats, default=1
        Output spacing :math:`(s_w, s_h, s_d)` in mm. If only one value
        :math:`s` is specified, then :math:`s_w = s_h = s_d = s`.

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
            - `gaussian`: Gaussian interpolation
              (:math:`\\sigma=0.8,\\alpha=4`).
            - `label_gaussian`: Gaussian interpolation for label maps
              (:math:`\\sigma = 1, \\alpha = 1`).
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

    Examples
    --------
    >>> import numpy as np
    >>> from nidl.volume.transforms.preprocessing.spatial import Resample
    >>> # Create a dummy 3D image (e.g., shape: (128, 128, 128))
    >>> image = np.random.rand(128, 128, 128).astype(np.float32)
    >>> # Assume identity affine (voxel size 1mm in RAS)
    >>> affine = np.eye(4)
    >>> # Instantiate the transform to resample to 2mm isotropic
    >>> resampler = Resample(target=2.0, interpolation='linear')
    >>> # Apply the transform
    >>> resampled = resampler(image, affine)
    >>> print(resampled.shape)
    (64, 64, 64)
    >>> # Works the same with torch.Tensor
    >>> import torch
    >>> image_torch = torch.from_numpy(image)
    >>> resampled_torch = resampler(image_torch, affine)
    """

    def __init__(
        self,
        target: Union[float, tuple[float, float, float]] = 1.0,
        interpolation: str = "linear",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target = self._parse_spacing(target)
        self.interpolation = interpolation
        self.interpolator = Resample._parse_interpolation(interpolation)

    def apply_transform(
        self, data: TypeTransformInput, affine: Optional[np.ndarray] = None
    ) -> TypeTransformInput:
        """Resample the input data.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            The input data with shape :math:`(C, H, W, D)` or :math:`(H, W, D)`

        affine: np.ndarray of shape (4, 4) or None, default=None
            Affine transformation matrix of the input data in RAS format
            defining spacing/origin/direction of the input image (in mm).
            This is typically given by Nibabel in this format.
            If None, the identity matrix is used, assuming 1mm isotropic input
            spacing.

        Returns
        ----------
        data: np.ndarray or torch.Tensor
            Resampled data with shape :math:`(H', W', D')`  or
            :math:`(C, H', W', D')` and same type as input with
            :math:`H' = \\frac{s_h'}{s_h} H, W' = \\frac{s_w'}{s_w} W,
            D' = \\frac{s_d'}{s_d} D` where :math:`(s_h', s_w', s_d')`
            and :math:`(s_h, s_w, s_d)` are input and output spacing (in mm)
            respectively.

        """
        affine = self._check_affine_ras(affine)

        is_data_tensor = False
        if isinstance(data, torch.Tensor):  # Computations are performed on CPU
            is_data_tensor = True
            dtype, device = data.dtype, data.device
            data = data.detach().cpu().numpy()

        floating_sitk = Resample.as_sitk(data, affine)

        resampler = Stk.ResampleImageFilter()
        resampler.SetInterpolator(self.interpolator)
        reference_image = Resample.get_reference_image(
            floating_sitk,
            self.target,
        )
        resampler.SetReferenceImage(reference_image)
        resampled = resampler.Execute(floating_sitk)
        resampled = Resample.from_sitk(resampled, dim=data.ndim)

        if is_data_tensor:
            resampled = torch.as_tensor(resampled, dtype=dtype, device=device)
        return resampled

    @staticmethod
    def as_sitk(data: np.ndarray, affine: np.ndarray) -> Stk.Image:
        """Convert the input data to a SimpleITK image."""
        is_multidim = data.ndim == 4
        image = Stk.GetImageFromArray(data.transpose(), isVector=is_multidim)

        origin, spacing, direction = (
            Resample.get_sitk_metadata_from_ras_affine(affine)
        )
        image.SetOrigin(origin)  # should I add a 4th value if force_4d?
        image.SetSpacing(spacing)
        image.SetDirection(direction)

        num_spatial_dims = 3
        if data.ndim == 4:
            assert image.GetNumberOfComponentsPerPixel() == data.shape[0]
            assert image.GetSize() == data.shape[1 : 1 + num_spatial_dims]
        elif data.ndim == 3:
            assert image.GetNumberOfComponentsPerPixel() == 1
            assert image.GetSize() == data.shape[:num_spatial_dims]
        else:  # should never happen
            raise ValueError(
                f"Input data must have 3 or 4 dimensions, got {data.ndim}"
            )
        return image

    @staticmethod
    def from_sitk(image: Stk.Image, dim: int) -> np.ndarray:
        """Convert the SimpleITK image as numpy array."""
        data = Stk.GetArrayFromImage(image).transpose()
        num_components = image.GetNumberOfComponentsPerPixel()
        if dim == 3:
            assert num_components == 1
            return data
        else:  # 4d
            if num_components == 1:
                data = data[np.newaxis]  # add channels dimension
            assert num_components == data.shape[0]
            return data

    @staticmethod
    def get_sitk_metadata_from_ras_affine(affine: np.ndarray):
        """Get the metadata from the affine matrix in LPS format (ITK) from RAS
        format (Nibabel)."""
        # Matrix used to switch between LPS and RAS
        FLIPXY_33 = np.diag([-1, -1, 1])
        direction_ras, spacing_array = (
            Resample.get_rotation_and_spacing_from_affine(affine)
        )
        origin_ras = affine[:3, 3]
        origin_lps = np.dot(FLIPXY_33, origin_ras)
        direction_lps = np.dot(FLIPXY_33, direction_ras)
        return origin_lps, spacing_array, direction_lps.flatten()

    @staticmethod
    def get_rotation_and_spacing_from_affine(
        affine: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # From https://github.com/nipy/nibabel/blob/master/nibabel/orientations.py
        rotation_zoom = affine[:3, :3]
        spacing = np.sqrt(np.sum(rotation_zoom * rotation_zoom, axis=0))
        rotation = rotation_zoom / spacing
        return rotation, spacing

    @staticmethod
    def get_reference_image(
        floating_sitk: Stk.Image, spacing: tuple[float, float, float]
    ) -> Stk.Image:
        old_spacing = np.array(floating_sitk.GetSpacing())
        new_spacing = np.array(spacing)
        old_size = np.array(floating_sitk.GetSize())
        new_size = old_size * old_spacing / new_spacing
        new_size = np.ceil(new_size).astype(np.uint16)
        new_size[old_size == 1] = 1  # keep singleton dimensions
        new_origin_index = 0.5 * (new_spacing / old_spacing - 1)
        new_origin_lps = floating_sitk.TransformContinuousIndexToPhysicalPoint(
            new_origin_index,
        )
        reference = Stk.Image(
            new_size.tolist(),
            floating_sitk.GetPixelID(),
            floating_sitk.GetNumberOfComponentsPerPixel(),
        )
        reference.SetDirection(floating_sitk.GetDirection())
        reference.SetSpacing(new_spacing.tolist())
        reference.SetOrigin(new_origin_lps)
        return reference

    @staticmethod
    def _parse_interpolation(interpolation):
        if interpolation == "nearest":
            return Stk.sitkNearestNeighbor
        elif interpolation == "linear":
            return Stk.sitkLinear
        elif interpolation == "bspline" or interpolation == "cubic":
            return Stk.sitkBSpline
        elif interpolation == "gaussian":
            return Stk.sitkGaussian
        elif interpolation == "label_gaussian":
            return Stk.sitkLabelGaussian
        elif interpolation == "hamming":
            return Stk.sitkHammingWindowedSinc
        elif interpolation == "cosine":
            return Stk.sitkCosineWindowedSinc
        elif interpolation == "welch":
            return Stk.sitkWelchWindowedSinc
        elif interpolation == "lanczos":
            return Stk.sitkLanczosWindowedSinc
        elif interpolation == "blackman":
            return Stk.sitkBlackmanWindowedSinc
        else:
            message = (
                f'Interpolation method "{interpolation}" not recognized.'
                " Please use one of the following: nearest, linear, bspline, "
                "cubic, gaussian, label_gaussian, hamming, cosine, welch, "
                "lanczos, blackman"
            )
            raise ValueError(message)

    @staticmethod
    def _parse_spacing(spacing):
        result: Iterable
        if isinstance(spacing, Iterable) and len(spacing) == 3:
            result = spacing
        elif isinstance(spacing, Number):
            result = 3 * (spacing,)
        else:
            message = (
                "Target must be a positive number"
                f" or a sequence of 3 positive numbers, not {type(spacing)}"
            )
            raise ValueError(message)
        if np.any(np.array(spacing) <= 0):
            message = f'Spacing must be strictly positive, not "{spacing}"'
            raise ValueError(message)
        return result

    @staticmethod
    def _check_affine_ras(affine):
        if affine is None:
            affine = np.eye(4)
        # Check type
        affine = check_array(affine, ensure_2d=True)
        # Check shape
        if affine.shape != (4, 4):
            raise ValueError("Affine must be 4x4 matrix")
        # Check orientation
        orientation = aff2axcodes(affine)
        if orientation != ("R", "A", "S"):
            raise ValueError(
                f"Affine is not in RAS orientation, got {orientation}"
            )
        return affine
