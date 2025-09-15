##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from typing import Union

import numpy as np
import torch

from ...volume_transform import TypeTransformInput, VolumeTransform


class CropOrPad(VolumeTransform):
    """Crop and/or pad a 3d volume to match the target shape.
    
    It handles :class:`np.ndarray` or :class:`torch.Tensor` as input and
    returns a consistent output (same type).

    Parameters
    ----------
    target_shape: int or tuple[int, int, int]
        Expected output shape. If int, apply the same size across all
        dimensions.

    padding_mode: str in {'edge', 'maximum', 'constant', 'mean', 'median',\
        'minimum', 'reflect', 'symmetric'}
        Possible modes for padding. See more infos in the `Numpy documentation
        <https://numpy.org/doc/stable/reference/generated/numpy.pad.html>`_.
        
    constant_values: float or tuple[float, float, float]
        The values to set the padded values for each
        axis if the padding mode is 'constant'.

    kwargs: dict
        Keyword arguments given to :class:`nidl.transforms.Transform`.

    """

    def __init__(
        self,
        target_shape: Union[int, tuple[int, int, int]],
        padding_mode: str = "constant",
        constant_values: Union[float, tuple[float, float, float]] = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.target_shape = self._parse_shape(target_shape, length=3)
        self.padding_mode = padding_mode
        self.constant_values = constant_values

    def apply_transform(self, data: TypeTransformInput) -> TypeTransformInput:
        """Crop and/or pad the input data to match target shape.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            The input data with shape :math:`(C, H, W, D)` or
            :math:`(H, W, D)`. Transformation is applied across
            all channels.

        Returns
        ----------
        data: np.ndarray or torch.Tensor
            Cropped or padded data with same type as input and
            shape `target_shape`.

        """
        crop_bounding_box = []
        pad_widths = []
        in_shape = data.shape
        out_shape = self.target_shape

        if data.ndim == 4:  # ignore channel dimension
            in_shape = in_shape[1:]
            crop_bounding_box.append(slice(None))
            pad_widths.append((0, 0))

        if isinstance(out_shape, int):
            out_shape = 3 * (out_shape,)

        if len(out_shape) != len(in_shape):
            raise ValueError(
                "'target_shape' should have same length as input "
                f"dimension, got {len(out_shape)} != {len(in_shape)}"
            )
        for dim in range(3):
            if out_shape[dim] >= in_shape[dim]:  # pad, no crop
                crop_bounding_box.append(slice(0, in_shape[dim]))
                pad_before = (out_shape[dim] - in_shape[dim]) // 2
                pad_after = max(
                    out_shape[dim] - (in_shape[dim] + pad_before), 0
                )
                pad_widths.append((pad_before, pad_after))
            else:  # crop, no pad
                crop_from = (in_shape[dim] - out_shape[dim]) // 2
                crop_until = crop_from + out_shape[dim]
                crop_bounding_box.append(slice(crop_from, crop_until))
                pad_widths.append((0, 0))

        # First crop the volume
        data = data[tuple(crop_bounding_box)]

        # Then pad
        is_tensor = False
        if isinstance(data, torch.Tensor):
            # convert to numpy for padding
            is_tensor = True
            dtype, device = data.dtype, data.device
            data = data.detach().cpu().numpy()

        kwargs_pad = {}
        if self.padding_mode == "constant":
            kwargs_pad.update(constant_values=self.constant_values)

        data = np.pad(data, pad_widths, mode=self.padding_mode, **kwargs_pad)

        if is_tensor:
            data = torch.as_tensor(data, dtype=dtype, device=device)

        return data
