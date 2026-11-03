##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn.functional as F  # ruff: ignore[lowercase-imported-as-non-lowercase]

from ....transforms import TypeTransformInput, VolumeTransform
from ._utils import standardize_foreground, to_float_volume


class BrainIACPreprocessing(VolumeTransform):
    r"""Preprocessing expected by the BrainIAC weights.

    BrainIAC was pretrained on brain MRI (T1-weighted, T2-weighted, FLAIR,
    ...) prepared by its authors as follows: N4 bias-field correction, rigid
    registration to their template and skull-stripping with HD-BET. The
    template is `temp_head.nii.gz`, provided with the `BrainIAC code
    <https://github.com/AIM-KannLab/BrainIAC>`_: a
    :math:`170 \times 206 \times 162` volume at 1 mm, in the LAS orientation.

    This transform turns such a volume into the input of the model:

    1) optionally, it crops the volume to the field of view of the template
       (see `crop`);
    2) it resizes the whole volume to :math:`96 \times 96 \times 96`;
    3) it standardizes the brain voxels (the non-zero ones) to zero mean and
       unit variance, the background staying at zero.

    Parameters
    ----------
    crop : sequence of (int, int) or None, default=None
        Start and end indices on each axis, to crop a volume registered to
        another template to the field of view of the BrainIAC template. As
        the whole volume is resized, the field of view sets the size of the
        brain in the model input. For OpenBHB quasi-raw volumes
        (:math:`182 \times 218 \times 182`, 1 mm, LAS), use
        ``((6, 176), (7, 213), (3, 165))``. If None, the volume is not
        cropped.
    **kwargs : dict
        Keyword arguments given to :class:`nidl.transforms.Transform`.

    Notes
    -----
    Expected input: a skull-stripped and bias-corrected brain MRI of shape
    :math:`(C, H, W, D)` or :math:`(H, W, D)`, either on the BrainIAC template
    grid, or at 1 mm in the LAS orientation with `crop` matching its template.
    The volume is neither reoriented nor resampled.

    Examples
    --------
    >>> import numpy as np
    >>> transform = BrainIACPreprocessing()
    >>> x = np.random.rand(1, 170, 206, 162)
    >>> transform(x).shape
    torch.Size([1, 96, 96, 96])
    """

    OUTPUT_SHAPE = (96, 96, 96)

    def __init__(
        self, crop: Optional[Sequence[tuple[int, int]]] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.crop = crop

    def apply_transform(self, data: TypeTransformInput) -> torch.Tensor:
        """Preprocess a volume.

        Parameters
        ----------
        data : np.ndarray or torch.Tensor
            Volume of shape :math:`(C, H, W, D)` or :math:`(H, W, D)`.

        Returns
        -------
        output : torch.Tensor
            Float32 volume of shape :math:`(C, 96, 96, 96)`.
        """
        x = torch.from_numpy(to_float_volume(data))
        for axis, (start, end) in enumerate(self.crop or (), start=1):
            x = x.narrow(axis, start, end - start)
        x = F.interpolate(
            x.unsqueeze(0), size=self.OUTPUT_SHAPE, mode="trilinear"
        ).squeeze(0)
        return standardize_foreground(x)
