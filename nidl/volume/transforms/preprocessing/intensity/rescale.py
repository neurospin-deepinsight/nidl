##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from typing import Callable, Optional

import numpy as np
import torch

from .....transforms import TypeTransformInput, VolumeTransform


class RobustRescaling(VolumeTransform):
    """Rescale intensities in a 3d volume to a given range.

    It is robust to outliers since the volume is clipped according to
    a given inter-quantile range. It applies the following percentile-based
    min-max transformation per channel:

    .. math::

        x_i' = \\frac{\\min\\left(\\max\\left(x_i, p_l\\right), p_u\\right) -
        p_l}{p_u - p_l} (o_{max} - o_{min}) + o_{min}

    .. math::

        p_{l} = \\text{percentile}(x, p_{min}), \\quad
        p_{u} = \\text{percentile}(x, p_{max})

    where :math:x_i is the original voxel intensity, :math:`(p_{\text{min}},
    p_{\text{max}})` defines the input quantile range used for clipping, and
    :math:`(o_{\text{min}}, o_{\text{max}})` defines the target output
    intensity range.

    It handles a :class:`numpy.ndarray` or :class:`torch.Tensor` as input and
    returns a consistent output (same type and shape). Input shape must be
    :math:`(C, H, W, D)` or :math:`(H, W, D)` (spatial dimensions).


    Parameters
    ----------
    out_min_max: (float, float), default=(0, 1)
        Range of output intensities.

    percentiles: (float, float), default=(1, 99)
        Percentage for the quantile values of the input volume used to clip
        the data. For example, SynthSeg [1]_ uses (1, 99) while nnUNet [2]_
        uses (0.5, 99.5).

    masking_fn: Callable or None, default=None
        If Callable, a masking function returning a boolean mask to be applied
        on the input volume for each channel separately. Only voxels inside the
        mask are used to compute the cutoff values when clipping the data. If
        None, the whole volume is taken to compute the cutoff.

    kwargs: dict
        Keyword arguments given to :class:`nidl.transforms.Transform`.

    Notes
    -----
    If the input volume has constant values, the normalized output is set to
    its minimum value by convention.

    References
    ----------
    .. [1] Billot, B. et al., (2023). "SynthSeg: Segmentation of brain MRI
           scans of any contrast and resolution without retraining."
           Medical Image Analysis, 86,  102789.
    .. [2] Isensee, F. et al., (2021). "nnU-Net: a self-configuring method
           for deep learning-based biomedical image segmentation."
           Nature Methods, 18, 203-211.

    Examples
    --------
    >>> import numpy as np
    >>> from nidl.volume.transforms import RobustRescaling
    >>> # Create a random 3d volume with shape (64, 64, 64)
    >>> volume = np.random.normal(loc=100, scale=20, size=(64, 64, 64))
    >>> # Define the transform
    >>> transform = RobustRescaling(out_min_max=(0, 1), percentiles=(1, 99))
    >>> # Apply the transform
    >>> rescaled = transform(volume)
    >>> rescaled.shape
    (64, 64, 64)
    >>> # Values are now in the range [0, 1]
    >>> rescaled.min(), rescaled.max()
    (0.0, 1.0)
    """

    def __init__(
        self,
        out_min_max: tuple[float, float] = (0, 1),
        percentiles: tuple[float, float] = (1, 99),
        masking_fn: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.out_min_max = self._parse_range(out_min_max)
        self.percentiles = self._parse_range(
            percentiles, check_min=0, check_max=100
        )
        self.masking_fn = masking_fn

    def apply_transform(self, data: TypeTransformInput) -> TypeTransformInput:
        """Apply the intensity rescaling.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            The input data with shape :math:`(C, H, W, D)` or :math:`(H, W, D)`

        Returns
        -------
        array or torch.Tensor
            The rescaled data with same type as input.
        """

        def _clip(data, mask, percentiles):
            masked_data = data[mask] if mask is not None else data
            cutoff = np.percentile(masked_data, percentiles).astype(data.dtype)
            return np.clip(data, *cutoff)

        def _rescale(data, out_min, out_max):  # not in-place
            out_range = out_max - out_min
            in_min, in_max = np.min(data), np.max(data)
            # if data is constant: set it to minimum value.
            if np.abs(in_max - in_min) < 1e-8:
                output = np.ones_like(data) * out_min
            else:
                output = (data - in_min) / (in_max - in_min)
                output = output * out_range + out_min
            return output

        is_data_tensor = isinstance(data, torch.Tensor)
        if is_data_tensor:
            device = data.device
            dtype = data.dtype
            data = data.detach().cpu().numpy()
        output = np.zeros_like(data, dtype=data.dtype)

        if data.ndim == 3:
            mask = (
                self.masking_fn(data) if self.masking_fn is not None else None
            )
            output = _clip(data, mask, self.percentiles)
            output = _rescale(output, *self.out_min_max)
        else:  # apply per channel
            for c in range(data.shape[0]):
                mask = (
                    self.masking_fn(data[c])
                    if self.masking_fn is not None
                    else None
                )
                output[c] = _clip(data[c], mask, self.percentiles)
                output = _rescale(output, *self.out_min_max)

        if is_data_tensor:
            output = torch.as_tensor(output, dtype=dtype, device=device)
        return output
