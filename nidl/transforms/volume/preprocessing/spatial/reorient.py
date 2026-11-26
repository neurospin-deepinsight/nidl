##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

import nibabel.orientations as nio
import numpy as np
import torch

from ....transforms import TypeTransformInput, VolumeTransform


def _parse_orientation(orientation: str) -> tuple[str, ...]:
    codes = tuple(orientation.upper())
    try:
        ornt = nio.axcodes2ornt(codes)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid orientation '{orientation}'.") from e
    if len(codes) != 3 or np.isnan(ornt).any():
        raise ValueError(f"Invalid orientation '{orientation}'.")
    return codes


class Reorient(VolumeTransform):
    """Reorient a 3d volume, by flipping and permuting its axes.

    Orientations are given as three axis codes, following the `nibabel
    convention <https://nipy.org/nibabel/coordinate_systems.html>`_: "RAS"
    means that the first axis goes from left to Right, the second from
    posterior to Anterior and the third from inferior to Superior. No
    interpolation is done.

    It handles :class:`numpy.ndarray` or :class:`torch.Tensor` as input and
    returns a consistent output (same type).

    Parameters
    ----------
    input_orientation : str
        Orientation of the input volumes, e.g. "LAS". Arrays do not carry this
        information and a wrong value silently mirrors the brain, hence there
        is no default.
    target_orientation : str, default="RAS"
        Orientation of the output volumes.
    kwargs : dict
        Keyword arguments given to
        :class:`nidl.transforms.transforms.Transform`.

    Examples
    --------
    >>> import numpy as np
    >>> from nidl.transforms.volume.preprocessing import Reorient
    >>> volume = np.random.rand(182, 218, 182)
    >>> Reorient("ASR")(volume).shape
    (182, 182, 218)
    """

    def __init__(
        self,
        input_orientation: str,
        target_orientation: str = "RAS",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_orientation = "".join(_parse_orientation(input_orientation))
        self.target_orientation = "".join(
            _parse_orientation(target_orientation)
        )
        self.ornt = nio.ornt_transform(
            nio.axcodes2ornt(tuple(self.input_orientation)),
            nio.axcodes2ornt(tuple(self.target_orientation)),
        )

    def apply_transform(self, data: TypeTransformInput) -> TypeTransformInput:
        """Reorient the input data.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            The input data with shape :math:`(C, H, W, D)` or
            :math:`(H, W, D)`. All channels are reoriented the same way.

        Returns
        -------
        data: np.ndarray or torch.Tensor
            Reoriented data with same type as input.
        """
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            dtype, device = data.dtype, data.device
            data = data.detach().cpu().numpy()
        if data.ndim == 4:
            output = np.stack(
                [nio.apply_orientation(c, self.ornt) for c in data]
            )
        else:
            output = nio.apply_orientation(data, self.ornt)
        output = np.ascontiguousarray(output)
        if is_tensor:
            output = torch.as_tensor(output, dtype=dtype, device=device)
        return output
