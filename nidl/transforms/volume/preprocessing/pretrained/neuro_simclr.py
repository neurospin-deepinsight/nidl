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

import numpy as np
import torch

from ....transforms import TypeTransformInput, VolumeTransform
from ..spatial import Reorient
from ._utils import standardize_foreground, to_float_volume


class NeuroSimCLRPreprocessing(VolumeTransform):
    r"""Preprocessing expected by the Neuro-SimCLR weights.

    Neuro-SimCLR was pretrained on T1-weighted MRI prepared by its authors
    with `TurboPrep <https://github.com/LemuelPuglisi/turboprep>`_: rigid
    registration to the MNI152 ICBM 2009c symmetric template
    (:math:`193 \times 229 \times 193` at 1 mm, RAS), intensity normalization
    and skull-stripping.

    This transform turns such a volume into the input of the model, as the
    official code does:

    1) it reorients the volume to RAS and optionally pads it to the template
       grid (see `padding`);
    2) it reverses the order of the axes and crops the volume to
       :math:`150 \times 192 \times 192`;
    3) it standardizes the brain voxels (the non-zero ones) to zero mean and
       unit variance, the background staying at zero.

    Parameters
    ----------
    input_orientation : str
        Orientation of the input volumes, as three axis codes, e.g. "RAS" or
        "LAS" (OpenBHB volumes).
    padding : sequence of (int, int) or None, default=None
        Zero-padding before and after each axis, applied after the
        reorientation, to bring a volume registered to another 1 mm template
        to the grid of the ICBM 2009c template. For OpenBHB quasi-raw volumes
        (:math:`182 \times 218 \times 182`), use ``((5, 6), (6, 5), (6, 5))``.
    **kwargs : dict
        Keyword arguments given to :class:`nidl.transforms.Transform`.

    Notes
    -----
    Expected input: a skull-stripped brain MRI on the grid of the ICBM 2009c
    template, or on another 1 mm grid with `padding` matching it, of shape
    :math:`(C, H, W, D)` or :math:`(H, W, D)`.
    """

    TEMPLATE_SHAPE = (193, 229, 193)
    OUTPUT_SHAPE = (150, 192, 192)

    def __init__(
        self,
        input_orientation: str,
        padding: Optional[Sequence[tuple[int, int]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reorient = Reorient(input_orientation, "RAS")
        self.input_orientation = self.reorient.input_orientation
        self.padding = padding

    def apply_transform(self, data: TypeTransformInput) -> torch.Tensor:
        """Preprocess a volume.

        Parameters
        ----------
        data : np.ndarray or torch.Tensor
            Volume of shape :math:`(C, H, W, D)` or :math:`(H, W, D)`.

        Returns
        -------
        output : torch.Tensor
            Float32 volume of shape :math:`(C, 150, 192, 192)`.
        """
        volume = self.reorient(to_float_volume(data))
        if self.padding is not None:
            volume = np.pad(volume, ((0, 0), *self.padding))
        if volume.shape[1:] != self.TEMPLATE_SHAPE:
            raise ValueError(
                f"Expected volumes on the {self.TEMPLATE_SHAPE} grid of the "
                f"template, got {volume.shape[1:]}: see `padding`."
            )
        x = torch.from_numpy(volume).permute(0, 3, 2, 1)
        return standardize_foreground(x[:, 10:160, 19:211, 1:])
