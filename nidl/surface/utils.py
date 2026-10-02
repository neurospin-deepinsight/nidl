##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Surfacic utilities.
"""

import numpy as np
from scipy.interpolate import griddata


def text2grid(
        vertices: np.ndarray,
        texture: np.ndarray,
        resx: int = 192,
        resy: int = 192
    ) ->  np.ndarray:
    """
    Convert a texture onto a spherical surface into an image by evenly
    resampling the spherical surface with respect to sin(e) and a, where e
    and a are elevation and azimuth, respectively. Nearest-neighbor
    interpolation is used to convert data from the 3-D surface to the
    2-D grid.

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)
        x, y, z coordinates of an icosahedron.
    texture : np.ndarray, shape (N, )
        Input icosahedron texture.
    resx : int
        Generated image number of samples in the x direction.
        Default 192.
    resy : int
        Generated image number of samples in the y direction.
        Default 192.

    Returns
    -------
    np.ndarray, shape (resx, resy)
        Projected texture.

    Examples
    --------
    >>> import numpy as np
    >>> from nidl.surface.base import Ico
    >>> from nidl.surface.utils import text2grid
    >>> verts, tris = Ico.mesh(order=2)
    >>> texture = np.ones((len(verts), ))
    >>> texture_arr = text2grid(verts, texture)
    >>> texture_arr.shape
    (192, 192)
    """
    azimuth, elevation, _ = cart2sph(*vertices.T)
    points = np.stack((azimuth, np.sin(elevation))).T
    grid_x, grid_y = np.mgrid[-np.pi:np.pi:resx * 1j, -1:1:resy * 1j]
    return griddata(points, texture, (grid_x, grid_y), method="nearest")


def cart2sph(
        x: float | np.ndarray,
        y: float | np.ndarray,
        z: float | np.ndarray,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Cartesian to spherical coordinate transform.

    Parameters
    ----------
    x: float | np.ndarray
        x-component of Cartesian coordinates
    y: float | np.ndarray
        y-component of Cartesian coordinates
    z: float | np.ndarray
        z-component of Cartesian coordinates

    Returns
    -------
    alpha: float | np.ndarray
        Azimuth angle in radiants. The value of the angle is in the range
        [-pi pi].
    beta: float | np.ndarray
        Elevation angle in radiants. The value of the angle is in the range
        [-pi/2, pi/2].
    r: float | np.ndarray
        Radius.
    """
    alpha = np.arctan2(y, x)
    beta = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return alpha, beta, r
