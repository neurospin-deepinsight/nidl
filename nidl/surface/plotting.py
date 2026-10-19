##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Surface plotting functions.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D


class IcoRenderer:
    """
    Simple 3D renderer for icosahedral meshes using Matplotlib.

    This class provides utilities to visualize triangular meshes (e.g.,
    icosahedral subdivisions), optionally with per-face textures, and to
    overlay points on the surface. The renderer hides axes, panes, and ticks
    to produce clean, publication-ready spherical visualizations.

    Parameters
    ----------
    figsize : tuple[int]
        Size of the Matplotlib figure.
        Default is (10, 10).
    fig : plt.Figure | None
        An existing Matplotlib figure to draw into. If None, a new figure
        is created.
        Default None.
    ax : Axes3D | None
        An existing 3D axis to draw into. If None, a new 3D axis is created.
        Default None.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import numpy as np
    >>> from nidl.surface.plotting import IcoRenderer
    >>> vertices = np.array([
    ...     [0, 0, 1],
    ...     [1, 0, 0],
    ...     [0, 1, 0],
    ... ])
    >>> triangles = np.array([[0, 1, 2]])
    >>> texture = np.array([0.2, 0.8, 0.5])
    >>> renderer = IcoRenderer(figsize=(8, 8))
    >>> renderer.add_trisurf(vertices, triangles)
    >>> renderer.add_texture(texture)
    >>> renderer.add_points(np.array([[1., 0., 0.]]), color="red", size=50)
    >>> renderer.show()
    """
    def __init__(
            self,
            figsize: tuple[int] = (10, 10),
            fig: plt.Figure | None = None,
            ax: Axes3D | None = None
        ) -> None:
        # Create figure
        if fig is None or ax is None:
            self.fig = plt.figure(figsize=figsize)
            self.ax = self.fig.add_subplot(111, projection="3d", aspect="auto")
        else:
            self.fig = fig
            self.ax = ax
        self.tri = None

        # Size
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)

        # Remove panes ans spines
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.set_pane_color((1, 1, 1, 0))
            axis.line.set_color((1, 1, 1, 0))

        # Remove ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])

    def add_trisurf(
            self,
            vertices: np.ndarray,
            triangles: np.ndarray,
        ) -> None:
        """
        Add a triangular surface mesh.

        Parameters
        ----------
        vertices : np.ndarray, shape (N, 3)
            Array of vertex coordinates.
        triangles : np.ndarray, shape (M, 3)
            Array of triangular faces.
        """
        # Unpack vertices
        x, y, z = vertices.T

        # Draw trisurf
        self.tri = self.ax.plot_trisurf(
            x, y, z,
            triangles=triangles,
            color="white",
            edgecolor="black",
            linewidth=0.3,
            alpha=0.1,
        )
        self.tri.triangles = triangles

    def add_texture(
            self,
            texture: np.ndarray,
            vmin: float | None = None,
            vmax: float | None = None,
            is_label: bool = False,
            cmap: str = "coolwarm") -> None:
        """
        Add a texture on a triangular surface mesh.

        Parameters
        ----------
        texture : np.ndarray, shape (N,)
            Per-vertex texture values. If provided, they are aggregated
            per face (mean or mode).
        vmin, vmax : float | None
            Value range for colormap.
        is_label : bool
            If True, texture is treated as integer labels and the mode
            of each triangle is used.
        cmap : str
            Color map used to display the texture.
            Default 'coolwarm'.

        Raises
        ------
        ValueError
            If a triangular surface is not already loaded.
        """
        # Checks
        if self.tri is None:
            raise ValueError(
                "You first need to add a triangular surface to the renderer."
            )

        # Auto vmin/vmax
        if vmin is None:
            vmin = texture.min()
        if vmax is None:
            vmax = texture.max()

        # Compute per-face texture if provided
        triangles = self.tri.triangles
        if is_label:
            # Mode of labels per triangle
            face_values = np.array([
                np.argmax(np.bincount(texture[tri].astype(int)))
                for tri in triangles
            ])
        else:
            # Mean value per triangle
            face_values = np.array([
                np.mean(texture[tri])
                for tri in triangles
            ])

        # Map values to colormap
        cmap = plt.get_cmap("coolwarm")
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        facecolors = cmap(norm(face_values))

        # Add texture
        self.tri.set_facecolors(facecolors)
        self.tri.set_alpha(1.)

    def add_points(
            self,
            points: np.ndarray,
            labels: list[str] | None = None,
            color: str = "red",
            size: int = 40,
        ) -> None:
        """
        Add points (dots) on the mesh, optionally with text labels.

        Parameters
        ----------
        points : np.ndarray, shape (K, 3)
            Coordinates of the points to display.
        labels : list[str] | None, shape (K, )
            Optional list of text labels for each point.
            Must have the same length as `points` if provided.
        color : str
            Color of the points.
        size : int
            Marker size.

        Raises
        ------
        ValueError
            If labels have not the same length as points.
        """
        points = np.asarray(points)

        # Draw points
        self.ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            marker="o",
            color=color,
            s=size,
        )

        # Draw labels (one per point)
        if labels is not None:
            if len(labels) != len(points):
                raise ValueError(
                    "Labels must have same length as points."
                )

            for (x, y, z), text in zip(points, labels):
                self.ax.text(
                    x, y, z,
                    text,
                    color=color,
                    size=(size / 5),
                )

    def show(
            self,
        ) -> None:
        """
        Display the current 3D figure.

        This renders all previously added surfaces, points, or overlays
        in the renderer. The method simply finalizes the layout and opens
        the Matplotlib window.
        """
        plt.tight_layout()
        plt.show()
