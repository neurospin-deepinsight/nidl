##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Spatial-based transformations.
"""

from typing import ClassVar

import numpy as np
from scipy.spatial import (
    cKDTree,
    transform,
)

from ..base import (
    Ico,
)
from .utils import (
    Interval,
    validate_data,
)


class RandomCutOut:
    """
    Cutout patches from input data with random parameters.

    Parameters
    ----------
    cuts: int | tuple[int, int]
        Number of cuts.
        Default 1.
    size: int | tuple[int, int]
        Number of neighboring rings used for the ablation.
        Default (1, 3).
    value : float
        Replacement patch value.
        Default 0.
    standard_ico : bool
        If True, recursively subdivides a standard icosahedron.
        If False, loads precomputed FreeSurfer icosahedra.
        Default False.

    Attributes
    ----------
    cuts_ : int
        Number of cuts. None if the sampler has not been called.
    sizes_ : list[int]
        Size of each cut. None if the sampler has not been called.

    Examples
    --------
    >>> import numpy as np
    >>> from nidl.surface.transforms import RandomCutOut
    >>> transform = RandomCutOut(cuts=1, size=(1, 3))
    >>> x = np.ones(642)
    >>> x_cutout = transform(x)
    >>> x_cutout.shape
    (642,)
    """
    bounds: ClassVar[dict[str, tuple[int, None, type]]] = {
        "cuts": (0, None, int),         # number of cuts must be >= 0
        "size": (0, None, int),         # number of rings must be >= 0
    }

    def __init__(
            self,
            cuts: int | tuple[int, int] = 1,
            size: int | tuple[int, int] = (1, 3),
            value: int = 0,
            standard_ico: bool = False,
        ):
        self.cuts_sampler = Interval(cuts, self.bounds["cuts"])
        self.size_sampler = Interval(size, self.bounds["size"])
        self.replacement_value = value
        self.standard_ico = standard_ico
        self.cuts_ = None
        self.sizes_ = None

    @validate_data(allowed_dims=(1, 2))
    def __call__(
            self,
            texture: np.ndarray,
        ) -> np.ndarray:
        """
        Cutout texture information.

        Parameters
        ----------
        texture : np.ndarray, shape (N,) or (C, N)
            Input texture.

        Returns
        -------
        np.ndarray
            Ablated texture (new array, not in-place).
        """
        self.cuts_ = self.cuts_sampler()
        self.sizes_ = [self.size_sampler() for _ in range(self.cuts_)]
        order = Ico.order(len(texture.T))

        crop_texture = texture.copy()
        for idx in range(self.cuts_):
            rand_v = np.random.randint(len(texture.T))
            neighs = Ico.neighbors(
                order,
                standard_ico=self.standard_ico,
                depth=self.sizes_[idx],
                direct_neighbor=True,
            )
            crop_texture[:, list(neighs[rand_v])] = self.replacement_value

        return crop_texture


class RandomRotation:
    """
    Rotate input data with random parameters.

    Parameters
    ----------
    phi : float | tuple[float, float]
        Rotation angle phi (in degrees) in Euler representation.
        If a tuple (low, high) is provided, the angle is sampled uniformly
        from [low, high].
        Default is (0., 5.).
    theta : float | tuple[float, float]
        Rotation angle theta (in degrees) in Euler representation.
        If a tuple (low, high) is provided, the angle is sampled uniformly
        from [low, high].
        Default is 0.
    psi : float | tuple[float, float]
        Rotation angle psi (in degrees) in Euler representation.
        If a tuple (low, high) is provided, the angle is sampled uniformly
        from [low, high].
        Default is 0.
    interpolation : str
        Interpolation used to rotate data: 'euclidian'.
        Default is 'euclidian'.
    standard_ico : bool
        If True, recursively subdivides a standard icosahedron.
        If False, loads precomputed FreeSurfer icosahedra.
        Default False.

    Attributes
    ----------
    phi_ : float
        Rotation angle phi (in degrees) in Euler representation.
    theta_ : float
        Rotation angle theta (in degrees) in Euler representation.
    psi_ : float
        Rotation angle psi (in degrees) in Euler representation.

    Examples
    --------
    >>> import numpy as np
    >>> from nidl.surface.transforms import RandomRotation
    >>> transform = RandomRotation(phi=(0., 5.), theta=0., psi=0.)
    >>> x = np.ones(642)
    >>> x_rotated = transform(x)
    >>> x_rotated.shape
    (642,)
    """
    bounds: ClassVar[dict[str, tuple[int, int, type]]] = {
        "phi": (0, 360, float),        # angle must be in [0, 360]
        "theta": (0, 360, float),      # angle must be in [0, 360]
        "psi": (0, 360, float),        # angle must be in [0, 360]
    }

    def __init__(
            self,
            phi: float | tuple[float, float] = (0., 5.),
            theta: float | tuple[float, float] = 0.0,
            psi: float | tuple[float, float] = 0.0,
            interpolation: str = "euclidian",
            standard_ico: bool = False,
        ) -> None:
        self.phi_sampler = Interval(phi, self.bounds["phi"])
        self.theta_sampler = Interval(theta, self.bounds["theta"])
        self.psi_sampler = Interval(psi, self.bounds["psi"])
        self.interpolation = interpolation
        self.standard_ico = standard_ico
        self.phi_ = None
        self.theta_ = None
        self.psi_ = None

    @validate_data(allowed_dims=(1, 2))
    def __call__(
            self,
            texture: np.ndarray,
        ) -> np.ndarray:
        """
        Rotate texture.

        Parameters
        ----------
        texture : np.ndarray, shape (N,) or (C, N)
            Input texture.

        Returns
        -------
        np.ndarray
            Rotated texture (new array, not in-place).
        """
        self.phi_ = self.phi_sampler()
        self.theta_ = self.theta_sampler()
        self.psi_ = self.psi_sampler()
        order = Ico.order(len(texture.T))

        neighs, weights = self.get_interpolation_coefficients(
            *Ico.mesh(order, standard_ico=self.standard_ico),
            [self.phi_, self.theta_, self.psi_],
            self.interpolation,
        )

        return self.interpolate_texture(texture, neighs, weights)

    @staticmethod
    def get_interpolation_coefficients(
            vertices: np.ndarray,
            triangles: np.ndarray,
            angles: tuple[float, float, float],
            interpolation: str = "euclidian",
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute interpolation coefficients after rotating an icosahedron.

        Parameters
        ----------
        vertices : np.ndarray, shape (N, 3)
            Original icosahedron vertices.
        triangles : np.ndarray, shape (M, 3)
            Triangle indices of the icosahedron.
        angles : tuple[float, float, float]
            Euler rotation angles in degrees.
        interpolation : str
            Interpolation used to rotate data: 'euclidian'.
            Default is 'euclidian'.

        Returns
        -------
        neighs : np.ndarray, shape (N, 3)
            Indices of the 3 neighbors used for interpolation.
        weights : anp.ndarray, shape (N, 3)
            Interpolation weights.

        Raises
        ------
        ValueError
            If an invalid interpolation type is specified.
        """
        # Rotate vertices
        rotation = transform.Rotation.from_euler("xyz", angles, degrees=True)
        rotated_vertices = rotation.apply(vertices)

        # Interpolation
        if interpolation == "euclidian":
            tree = cKDTree(rotated_vertices)
            dist, neighs = tree.query(vertices, k=3)
            weights = dist / dist.sum(axis=1, keepdims=True)
        else:  # TODO: barycentric interpolation
            raise ValueError(
                "Interpolation must be 'euclidian'."
            )

        return neighs, weights

    @staticmethod
    def interpolate_texture(
            texture: np.ndarray,
            neighs: np.ndarray,
            weights: np.ndarray,
        ) -> np.ndarray:
        """
        Interpolate a texture defined on a mesh using precomputed interpolation
        coefficients (neighbors and weights).

        It works for any texture dimensionality: scalar fields or multi-channel
        feature maps.

        Parameters
        ----------
        texture : np.ndarray, shape (C, N)
            Texture values defined on the original mesh.
            - N is the number of vertices.
            - C is the number of channels (1 for scalar, 3 for RGB, etc.).
        neighs : np.ndarray, shape (N, 3)
            Indices of the 3 neighbors used for interpolation.
        weights : anp.ndarray, shape (N, 3)
            Interpolation weights.

        Returns
        -------
        np.ndarray, shape (N, ) or (C, N)
            The interpolated texture. The output has the same number of
            vertices and channels as the input texture.

        Examples
        --------
        >>> import numpy as np
        >>> from nidl.surface.transforms import RandomRotation
        >>> texture = np.random.rand(3, 100)
        >>> neighs = np.random.randint(0, 100, size=(100, 3))
        >>> weights = np.random.rand(100, 3)
        >>> weights /= weights.sum(axis=1, keepdims=True)
        >>> interp = RandomRotation.interpolate_texture(
        ...     texture, neighs, weights
        ... )
        >>> interp.shape
        (3, 100)
        """
        # Gather the 3 neighbors for each vertex, shape (C, N, 3)
        gathered = texture[:, neighs]

        # Expand weights to broadcast over channels, shape (1, N, 3)
        weights = weights[None, ...]

        # Weighted sum, shape (C, N)
        return (gathered * weights).sum(axis=2)
