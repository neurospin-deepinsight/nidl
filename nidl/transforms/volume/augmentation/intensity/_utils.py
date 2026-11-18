##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

import numpy as np
import torch

from ....transforms import TypeTransformInput


def _to_float_numpy(data: TypeTransformInput) -> np.ndarray:
    """Floating point numpy copy of `data` (at least float32)."""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    return data.astype(np.result_type(data.dtype, np.float32))


def _like(out: np.ndarray, data: TypeTransformInput) -> TypeTransformInput:
    """`out` with the type, dtype and device of `data` (integers are
    rounded and clipped to the dtype range)."""
    dtype = data.dtype
    if isinstance(data, torch.Tensor):
        if not dtype.is_floating_point:
            info = torch.iinfo(dtype)
            out = np.clip(np.rint(out), info.min, info.max)
        return torch.as_tensor(out, dtype=dtype, device=data.device)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(dtype)
