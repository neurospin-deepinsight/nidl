from collections.abc import Iterable
from numbers import Number
from typing import Optional, Union, Tuple
import numpy as np
import SimpleITK as sitk
import torch
# Local import
from .....transform import Transform, TypeTransformInput


class Resample(Transform):
    """Resample 3D image to a different physical space.

    This is a powerful transform that can be used to change the image shape. 
    It uses SimpleITK as backend for efficiency.

    Code simplified from 
    https://github.com/TorchIO-project/torchio/blob/main/src/torchio/transforms/preprocessing/spatial/resample.py

    """

    def __init__(
        self,
        target: Union[float, Tuple[float, float, float]] = 1,
        interpolation: str = 'linear',
        **kwargs):
        """
        Parameters
        ----------

        target: float or tuple of floats, default=1
            Output spacing (s_w, s_h, s_d) in mm. If only one value s is specified, then s_w = s_h = s_d = s.

        interpolation: str in {'nearest', 'linear', 'bspline', 'cubic', 'gaussian', 'label_gaussian',
                'hamming', 'cosine', 'welch', 'lanczos', 'blackman'}, default='linear'
            
            Interpolation techniques available in ITK. 'linear' interpolation, the default in NeuroCLAV for 
            scalar images, is usually a good compromise between image quality and speed. It is therefore a 
            good choice for data augmentation during training. Methods such as 'bspline' or 'lanczos' generate 
            high-quality results, but are generally slower. They can be used to obtain optimal resampling 
            results during offline data preprocessing. 'nearest' can be used for quick experimentation as it is 
            very fast, but produces relatively poor results for scalar images. It is the default interpolation
            type for label maps, as categorical values for the different labels need to preserved after interpolation.
            
            For a full quantitative comparison of interpolation methods, you can read:
            `Meijering et al. 1999, Quantitative Comparison of Sinc-Approximating Kernels for Medical Image Interpolation`
            
            'nearest': Interpolates image intensity at a non-integer pixel position by copying the intensity for the nearest neighbor.
            
            'linear': Linearly interpolates image intensity at a non-integer pixel position.

            'bspline': B-Spline of order 3 (cubic) interpolation.
            
            'cubic': Same as 'bspline'

            'gaussian': Gaussian interpolation. Sigma is set to 0.8 input pixels and alpha is 4
            
            'label_gaussian': Smoothly interpolate multi-label images. Sigma is set to 1 input pixel and alpha is 1

            'hamming': Hamming windowed sinc kernel.

            'cosine': Cosine windowed sinc kernel.

            'welch': Welch windowed sinc kernel.

            'lanczos': Lanczos windowed sinc kernel.

            'blackman': Blackman windowed sinc kernel.


        **kwargs: See neuroclav.transforms.Transform for additional
            keyword arguments.
    
        """
        
        super().__init__(**kwargs)
        self.target = target
        self.interpolation = interpolation
        self.interpolator = Resample.parse_interpolation(
            interpolation
        )
    
    @staticmethod
    def parse_interpolation(interpolation):
        if interpolation == 'nearest':
            return sitk.sitkNearestNeighbor
        elif interpolation == 'linear':
            return sitk.sitkLinear
        elif interpolation == 'bspline':
            return sitk.sitkBSpline
        elif interpolation == 'cubic':
            return sitk.sitkBSpline
        elif interpolation == 'gaussian':
            return sitk.sitkGaussian
        elif interpolation == 'label_gaussian':
            return sitk.sitkLabelGaussian
        elif interpolation == 'hamming':
            return sitk.sitkHammingWindowedSinc
        elif interpolation == 'cosine':
            return sitk.sitkCosineWindowedSinc
        elif interpolation == 'welch':
            return sitk.sitkWelchWindowedSinc
        elif interpolation == 'lanczos':
            return sitk.sitkLanczosWindowedSinc
        elif interpolation == 'blackman':
            return sitk.sitkBlackmanWindowedSinc
        else:
            message = (
                f'Interpolation method "{interpolation}" not recognized.'
                ' Please use one of the following: nearest, linear, bspline, '
                'cubic, gaussian, label_gaussian, hamming, cosine, welch, lanczos, blackman'
            )
            raise ValueError(message)
        

    def parse_data(self, data: TypeTransformInput) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Unexpected type: %s"%type(data))
        

    @staticmethod
    def _parse_spacing(spacing) -> tuple[float, float, float]:
        result: Iterable
        if isinstance(spacing, Iterable) and len(spacing) == 3:
            result = spacing
        elif isinstance(spacing, Number):
            result = 3 * (spacing,)
        else:
            message = (
                'Target must be a positive number'
                f' or a sequence of 3 positive numbers, not {type(spacing)}'
            )
            raise ValueError(message)
        if np.any(np.array(spacing) <= 0):
            message = f'Spacing must be strictly positive, not "{spacing}"'
            raise ValueError(message)
        return result


    def apply_transform(self, data: np.ndarray, 
                        affine: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Parameters
        ----------
        data: array of shape (W, H, D)  or (*, W, H, D)
            Input data to resample.
        
        affine: array of shape (4, 4) or None, optional
            Affine transformation matrix of the input data in RAS format (Nibabel)
            defining spacing/origin/direction of the input image (in mm).
            If None, the identity matrix is used.
        
        Returns
        ----------
        array of shape (W', H', D')  or (*, W', H', D')
            Resampled data. If affine is None, then 
            W' = 1/s_w * W, H' = 1/s_h * H, D' = 1/s_d * D.
        """

        if data.ndim > 4 or data.ndim < 3:
            raise ValueError(
                'Input data must have 3 or 4 dimensions, not %iD' % data.ndim
            )

        floating_sitk = Resample.as_sitk(data, affine)

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(self.interpolator)
        target_spacing = self._parse_spacing(self.target)
        reference_image = Resample.get_reference_image(
            floating_sitk,
            target_spacing,
        )
        resampler.SetReferenceImage(reference_image)
        resampled = resampler.Execute(floating_sitk)
        resampled = Resample.from_sitk(resampled, dim=data.ndim)
        return resampled

    @staticmethod
    def as_sitk(data: np.ndarray, affine: Optional[np.ndarray] = None) -> sitk.Image:
        """Convert a numpy array to a SimpleITK image, assuming RAS format for affine transformation."""

        if affine is None:
            affine = np.eye(4)

        is_multidim = data.ndim == 4
        image = sitk.GetImageFromArray(data.transpose(), isVector=is_multidim)

        origin, spacing, direction = Resample.get_sitk_metadata_from_ras_affine(
            affine
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
        else:
            raise ValueError(
                'Input data must have 3 or 4 dimensions, not %iD' % data.ndim
            )
        return image
    
    @staticmethod
    def from_sitk(image: sitk.Image, dim: int) -> np.ndarray:
        data = sitk.GetArrayFromImage(image).transpose()
        num_components = image.GetNumberOfComponentsPerPixel()
        if dim == 3:
            assert num_components == 1
            return data
        elif dim == 4:
            if num_components == 1:
                data = data[np.newaxis]  # add channels dimension
            assert num_components == data.shape[0]
            return data
        else:
            raise ValueError(
                'Input data must have 3 or 4 dimensions, not %iD' % dim
            )

    @staticmethod
    def get_sitk_metadata_from_ras_affine(affine: np.ndarray):
        """Get the metadata from the affine matrix in LPS format (ITK) from RAS format (Nibabel)."""
        # Matrix used to switch between LPS and RAS
        FLIPXY_33 = np.diag([-1, -1, 1])
        direction_ras, spacing_array = Resample.get_rotation_and_spacing_from_affine(affine)
        origin_ras = affine[:3, 3]
        origin_lps = np.dot(FLIPXY_33, origin_ras)
        direction_lps = np.dot(FLIPXY_33, direction_ras)
        return origin_lps, spacing_array, direction_lps.flatten()
    
    @staticmethod
    def get_rotation_and_spacing_from_affine(affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # From https://github.com/nipy/nibabel/blob/master/nibabel/orientations.py
        rotation_zoom = affine[:3, :3]
        spacing = np.sqrt(np.sum(rotation_zoom * rotation_zoom, axis=0))
        rotation = rotation_zoom / spacing
        return rotation, spacing

    @staticmethod
    def get_reference_image(
        floating_sitk: sitk.Image,
        spacing: tuple[float, float, float]) -> sitk.Image:
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
        reference = sitk.Image(
            new_size.tolist(),
            floating_sitk.GetPixelID(),
            floating_sitk.GetNumberOfComponentsPerPixel(),
        )
        reference.SetDirection(floating_sitk.GetDirection())
        reference.SetSpacing(new_spacing.tolist())
        reference.SetOrigin(new_origin_lps)
        return reference
