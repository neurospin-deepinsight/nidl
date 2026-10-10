##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

import numbers
import random
from typing import Optional, Union

import numpy as np
import SimpleITK as Stk
import torch

from ....transforms import TypeTransformInput, VolumeTransform
from ...preprocessing.spatial.resample import Resample


class RandomAffine(VolumeTransform):
    r"""Apply a random affine transformation to a 3d volume.

    A single affine transformation (composing scaling, rotation, shearing
    and translation) is randomly sampled and applied to the spatial
    dimensions of the volume. Input shape must be
    :math:`(C, H, W, D)` or :math:`(H, W, D)`.

    Parameters
    ----------
    scale: (float, float), default=(0.9, 1.1)
        Scaling factor range. A single isotropic factor is sampled
        :math:`s \sim \mathcal{U}(a, b)` and shared across the three spatial
        axes (the volume keeps its aspect ratio). A factor of 1 leaves the
        scale unchanged. Must be strictly positive.
    degrees: float or (float, float), default=(-10, 10)
        Rotation in degrees. One angle per spatial axis is sampled
        :math:`\theta_i \sim \mathcal{U}(a, b)`. If a single number ``d`` is
        given, the symmetric range ``(-d, d)`` is used. Set to 0 to deactivate
        rotations.
    translation: float or (float, float, float), default=0.0
        Maximum absolute translation, given as a fraction of the image extent
        along each axis (3d generalization of torchvision's ``translate``).
        The shift along axis :math:`i` is sampled
        :math:`t_i \sim \mathcal{U}(-f_i L_i, f_i L_i)`, where :math:`f_i` is
        the fraction and :math:`L_i` the physical size (in mm) of that axis. A
        single float uses the same fraction for the three axes; a tuple gives
        one fraction per axis. Each fraction must lie in ``[0, 1]``. Defaults
        to no translation.
    shears: float or (float, float), default=0.0
        Shear factor (slope of the off-diagonal terms). One factor per
        off-diagonal entry (six in total) is sampled
        :math:`h_i \sim \mathcal{U}(a, b)`. If a single number ``h`` is
        given, the symmetric range ``(-h, h)`` is used. A factor of 0 disables
        shearing.
    interpolation: str in {'nearest', 'linear', 'bspline', 'cubic', \
        'gaussian', 'label_gaussian', 'hamming', 'cosine', 'welch', \
        'lanczos', 'blackman'}, default='linear'
        Interpolation technique available in ITK. `linear` offers a good
        compromise between image quality and speed and is a solid choice for
        data augmentation during training. Use `nearest` for label maps to
        preserve categorical values. See
        :class:`~nidl.transforms.volume.preprocessing.spatial.Resample` for a
        description of every method.
    default_pad_value: float, default=0.0
        Value used to fill voxels mapped outside the input volume.
    kwargs: dict
        Keyword arguments given to :class:`nidl.transforms.Transform`
        (e.g. ``p``, the probability of applying the transform).

    Notes
    -----
    The transformation is centered on the volume: the image content is mapped
    according to :math:`y = M (x - c) + c + t`, where :math:`c` is the physical
    center of the volume, :math:`M = R \, H \, S` composes the sampled
    rotation :math:`R`, shear :math:`H` and scaling :math:`S`, and :math:`t` is
    the sampled translation. As SimpleITK's resampler pulls samples from the
    input (mapping output coordinates back into the input), the inverse of this
    transform is passed to it. The output keeps the same shape, spacing
    and orientation as the input.

    Examples
    --------
    >>> import torch
    >>> from nidl.transforms.volume.augmentation.spatial import RandomAffine
    >>> volume = torch.randn(1, 64, 64, 64)  # shape: (C, H, W, D)
    >>> transform = RandomAffine(scale=(0.9, 1.1), degrees=(-10, 10))
    >>> transformed = transform(volume)  # shape (1, 64, 64, 64)
    """

    def __init__(
        self,
        scale: tuple[float, float] = (0.9, 1.1),
        degrees: Union[float, tuple[float, float]] = (-10, 10),
        translation: Union[float, tuple[float, float, float]] = 0.0,
        shears: Union[float, tuple[float, float]] = 0.0,
        interpolation: str = "linear",
        default_pad_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale = self._parse_range(scale, check_min=0)
        if self.scale[0] <= 0:
            raise ValueError(f"`scale` must be strictly positive, got {scale}")
        self.degrees = self._parse_range_or_scalar(degrees)
        self.translation = self._parse_translation(translation)
        self.shears = self._parse_range_or_scalar(shears)
        self.interpolation = interpolation
        self.interpolator = Resample._parse_interpolation(interpolation)
        self.default_pad_value = default_pad_value

    def apply_transform(
        self,
        data: TypeTransformInput,
        affine: Optional[np.ndarray] = None,
    ) -> TypeTransformInput:
        """Apply a random affine transformation.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            Input volume of shape :math:`(C, H, W, D)` or :math:`(H, W, D)`.
            The same spatial transform is applied to every channel.

        affine: np.ndarray of shape (4, 4) or None, default=None
            Affine transformation matrix of the input data in RAS format
            defining spacing/origin/direction of the input image (in mm).
            This is typically given by Nibabel in this format. If None, the
            identity matrix is used, assuming 1mm isotropic spacing.

        Returns
        -------
        data: np.ndarray or torch.Tensor
            Transformed volume of same type and shape as input.
        """
        affine = Resample._check_affine_ras(affine)

        data_is_tensor = isinstance(data, torch.Tensor)
        if data_is_tensor:
            dtype, device = data.dtype, data.device
            data = data.detach().cpu().numpy()

        image = Resample.as_sitk(data, affine)
        transform = self._sample_transform(image)

        resampler = Stk.ResampleImageFilter()
        resampler.SetInterpolator(self.interpolator)
        resampler.SetReferenceImage(image)  # keep shape/spacing/orientation
        resampler.SetDefaultPixelValue(float(self.default_pad_value))
        resampler.SetTransform(transform.GetInverse())
        resampled = resampler.Execute(image)
        resampled = Resample.from_sitk(resampled, dim=data.ndim)

        if data_is_tensor:
            resampled = torch.as_tensor(resampled, dtype=dtype, device=device)
        return resampled

    def _sample_transform(self, image: Stk.Image) -> Stk.AffineTransform:
        """Sample a centered affine transform for the given image."""
        # A single isotropic factor is shared across the three spatial axes.
        scales = np.full(3, random.uniform(*self.scale))
        angles = np.deg2rad([random.uniform(*self.degrees) for _ in range(3)])
        shears = np.array([random.uniform(*self.shears) for _ in range(6)])

        # Compose the linear part. Applied to a point x as M @ x,
        # scale first, then shear, then rotate.
        matrix = (
            self._rotation_matrix(angles)
            @ self._shear_matrix(shears)
            @ np.diag(scales)
        )

        # Translation as a fraction of the physical extent of each axis
        size = np.array(image.GetSize())
        extent = size * np.array(image.GetSpacing())  # axis length in mm
        max_shift = np.array(self.translation) * extent
        translation = np.array([random.uniform(-m, m) for m in max_shift])

        # Center of the volume in physical (mm) coordinates so that the
        # rotation/scaling happen about the middle, not the corner.
        center_index = (size - 1) / 2.0
        center = image.TransformContinuousIndexToPhysicalPoint(
            center_index.tolist()
        )

        transform = Stk.AffineTransform(3)  # affine transform in 3d
        transform.SetCenter(center)
        transform.SetMatrix(matrix.ravel().tolist())
        transform.SetTranslation(translation.tolist())
        return transform

    @staticmethod
    def _rotation_matrix(angles: np.ndarray) -> np.ndarray:
        """Compose rotations (radians) around the x, y and z axes."""
        rx, ry, rz = angles
        cos, sin = np.cos, np.sin
        rot_x = np.array(
            [[1, 0, 0], [0, cos(rx), -sin(rx)], [0, sin(rx), cos(rx)]]
        )
        rot_y = np.array(
            [[cos(ry), 0, sin(ry)], [0, 1, 0], [-sin(ry), 0, cos(ry)]]
        )
        rot_z = np.array(
            [[cos(rz), -sin(rz), 0], [sin(rz), cos(rz), 0], [0, 0, 1]]
        )
        return rot_z @ rot_y @ rot_x

    @staticmethod
    def _shear_matrix(shears: np.ndarray) -> np.ndarray:
        """Build a shear matrix from the six off-diagonal factors."""
        s01, s02, s10, s12, s20, s21 = shears
        return np.array([[1, s01, s02], [s10, 1, s12], [s20, s21, 1]])

    def _parse_range_or_scalar(self, value, check_min=None, check_max=None):
        """Parse a ``(min, max)`` range, a scalar ``x`` becoming ``(-x, x)``.

        A single number :math:`x` is expanded to the symmetric range
        :math:`(-x, x)`.
        """
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    f"A single value must be non-negative, got {value}"
                )
            value = (-value, value)
        return self._parse_range(
            value, check_min=check_min, check_max=check_max
        )

    def _parse_translation(self, translation):
        """Parse the per-axis maximum absolute translation fractions.

        A scalar is broadcast to the three spatial axes; a sequence must give
        one fraction per axis. Each fraction must lie in ``[0, 1]``.
        """
        if isinstance(translation, numbers.Number):
            translation = (translation,) * 3
        translation = tuple(translation)
        if len(translation) != 3:
            raise ValueError(
                "translation must be a float or a sequence of three floats, "
                f"got {translation}"
            )
        for fraction in translation:
            if not isinstance(fraction, numbers.Number) or not (
                0 <= fraction <= 1
            ):
                raise ValueError(
                    "translation fractions must be numbers in [0, 1], got "
                    f"{translation}"
                )
        return translation
