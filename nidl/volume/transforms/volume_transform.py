##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from ...transforms import Transform, TypeTransformInput


class VolumeTransform(Transform):
    """Transformation applied to a 3d volume."""

    def parse_data(self, data: TypeTransformInput) -> TypeTransformInput:
        """Checks if the input data shape is 3d or 4d.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            The input data with shape :math:`(C, H, W, D)` or
            :math:`(H, W, D)`

        Returns
        ----------
        np.ndarray or torch.Tensor
            Data with type and shape checked.

        Raises
        ----------
        ValueError
            If the input data is not :class:`numpy.ndarray` or
            :class:`torch.Tensor` or if the shape is not 3d or 4d.
        """
        data = super().parse_data(data)

        if len(data.shape) not in [3, 4]:
            raise ValueError(
                "Input data must be 3d or 4d (channel+spatial dimensions), "
                f"got {len(data.shape)}"
            )
        return data
