##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # ruff: ignore[lowercase-imported-as-non-lowercase]

from ....transforms import TypeTransformInput, VolumeTransform
from ..intensity import RobustRescaling
from ..spatial import CropForeground, CropOrPad, Reorient
from ._utils import to_float_volume


class NeuroJEPAPreprocessing(VolumeTransform):
    r"""Preprocessing expected by the Neuro-JEPA weights.

    Neuro-JEPA was pretrained on brain MRI of several modalities prepared by
    its authors as follows: N4 bias-field correction, affine (12-DOF)
    registration to the MNI152 template at 1 mm and skull-stripping. The
    OpenBHB quasi-raw images are prepared in a similar way.

    This transform turns such a volume into the input of the model, as the
    official code does:

    1) it replaces NaN by 0 and reorients the volume to RAS;
    2) it rescales the intensities to :math:`[0, 1]`, the 0.5th and 99.5th
       percentiles of the whole volume being mapped to 0 and 1, and clips
       them;
    3) it crops the volume around the brain (the positive voxels), with a
       margin of 4 voxels, after bringing it to
       :math:`180 \times 216 \times 180`;
    4) it resizes the crop to :math:`100 \times 120 \times 100` and takes
       its center, of shape :math:`96 \times 108 \times 96`.

    The output is identical to the one of the official code.

    Parameters
    ----------
    input_orientation : str
        Orientation of the input volumes, as three axis codes: for instance
        "LAS" if the first axis goes from right to Left, the second from
        posterior to Anterior and the third from inferior to Superior.
        OpenBHB volumes are "LAS". Arrays do not carry this information and a
        wrong value silently mirrors the brain, hence there is no default.
    **kwargs : dict
        Keyword arguments given to :class:`nidl.transforms.Transform`.

    Notes
    -----
    Expected input: a skull-stripped and bias-corrected brain MRI, affinely
    registered to MNI152, at 1 mm isotropic resolution, of shape
    :math:`(C, H, W, D)` or :math:`(H, W, D)`. The volume is not resampled:
    it must already be at 1 mm.

    Examples
    --------
    >>> import numpy as np
    >>> transform = NeuroJEPAPreprocessing(input_orientation="LAS")
    >>> x = np.random.rand(1, 182, 218, 182)
    >>> transform(x).shape
    torch.Size([1, 96, 108, 96])
    """

    INTERMEDIATE_SHAPE = (180, 216, 180)
    RESIZED_SHAPE = (100, 120, 100)
    OUTPUT_SHAPE = (96, 108, 96)
    PERCENTILES = (0.5, 99.5)
    FOREGROUND_MARGIN = 4

    def __init__(self, input_orientation: str, **kwargs):
        super().__init__(**kwargs)
        self.reorient = Reorient(input_orientation, "RAS")
        self.input_orientation = self.reorient.input_orientation
        self.rescale = RobustRescaling(
            out_min_max=(0, 1), percentiles=self.PERCENTILES
        )
        self.pad = CropOrPad(self.INTERMEDIATE_SHAPE, padding_mode="edge")
        self.crop_foreground = CropForeground(margin=self.FOREGROUND_MARGIN)
        self.crop = CropOrPad(self.OUTPUT_SHAPE)

    def apply_transform(self, data: TypeTransformInput) -> torch.Tensor:
        """Preprocess a volume.

        Parameters
        ----------
        data : np.ndarray or torch.Tensor
            Volume of shape :math:`(C, H, W, D)` or :math:`(H, W, D)`, not
            modified.

        Returns
        -------
        output : torch.Tensor
            Float32 volume of shape :math:`(C, 96, 108, 96)`, with
            :math:`C=1` if the input has no channel.
        """
        volume = to_float_volume(data)
        volume[np.isnan(volume)] = 0.0
        volume = self.rescale(self.reorient(volume))
        volume = self.crop_foreground(self.pad(volume))
        x = F.interpolate(
            torch.from_numpy(np.ascontiguousarray(volume)).unsqueeze(0),
            size=self.RESIZED_SHAPE,
            mode="area",
        ).squeeze(0)
        return self.crop(x)
