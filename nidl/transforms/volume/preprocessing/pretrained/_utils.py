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

from ....transforms import TypeTransformInput


def to_float_volume(data: TypeTransformInput) -> np.ndarray:
    """Copy `data` to a float32 array of shape (C, H, W, D)."""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    volume = np.array(data, dtype=np.float32)
    return volume[np.newaxis] if volume.ndim == 3 else volume


def standardize_foreground(x: torch.Tensor) -> torch.Tensor:
    """Standardize `x` with the statistics of its non-zero voxels only.

    The background stays at zero. This differs from
    :class:`~nidl.transforms.volume.preprocessing.ZNormalization` with a
    mask, which also shifts the background and adds an epsilon to the
    standard deviation.
    """
    mask = x != 0
    if not mask.any():
        raise ValueError("Cannot standardize a volume without foreground.")
    values = x[mask]
    output = torch.zeros_like(x)
    output[mask] = (values - values.mean()) / values.std(unbiased=False)
    return output
