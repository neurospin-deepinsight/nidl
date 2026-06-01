##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Icosahedron structure.
"""

import functools
import inspect
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, ClassVar

import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors


class Ico:
    """
    Generate an icosahedral mesh of arbitrary subdivision order.

    This class includes an in-memory caching mechanism that stores the results
    of its public API methods to avoid recomputing identical meshes, patches,
    or neighbors and to significantly speed up repeated calls with the same
    parameters.
    """
    _cache: ClassVar[dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Caching methods
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(
            method_name: str,
            bound: dict[str, Any]) -> str:
        """
        Build a deterministic call-signature key of the form:

            method_name(param1=value1, param2=value2)

        This key is suitable for use in caches or logs where a stable,
        human-readable representation of a method invocation is required.

        The key includes:
        - the method name
        - all bound parameters (including defaults) except the 'n_jobs'
          parameter

        Parameters
        ----------
        method_name : str
            Name of the method being cached.
        bound : dict[str, Any]
            Mapping of parameter names to their bound values.

        Returns
        -------
        str
            A stable string representing the call signature.
        """

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (list, tuple)):
                return [convert(x) for x in obj]
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        args_repr = ", ".join(
            f"{k}={v!r}" for k, v in convert(bound).items()
            if k != "n_jobs"
        )
        return f"{method_name}({args_repr})"

    @staticmethod
    def cached(method: Callable) -> Callable:
        """
        Decorator that caches the result of a method based on its full
        signature.

        The cache key is generated using `_make_key`, which ensures that
        arguments of any type (including NumPy arrays) are converted into
        JSON-serializable structures.

        Parameters
        ----------
        method : Callable
            The method to wrap with caching.

        Returns
        -------
        Callable
            A wrapped version of the method that returns cached results
            when called with the same arguments.
        """

        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(method)
            args_for_binding = (
                args[1:]
                if (args and hasattr(method, "__qualname__") and
                    "." in method.__qualname__)
                else args
            )
            bound = sig.bind_partial(*args_for_binding, **kwargs)
            bound.apply_defaults()
            bound = dict(bound.arguments)

            key = Ico._make_key(method.__qualname__, bound)
            if key in Ico._cache:
                return Ico._cache[key]

            result = method(*args, **kwargs)
            Ico._cache[key] = result

            return result

        return wrapper

    @classmethod
    def clear(cls):
        """
        Clear the cache content.
        """
        cls._cache = {}

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    @classmethod
    @cached
    def mesh(
            cls,
            order: int = 3,
            standard_ico: bool = False,
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate the icosahedron mesh.

        Parameters
        ----------
        order : int
            Subdivision order of the icosahedron.
            Default 3.
        standard_ico : bool
            If True, recursively subdivides a standard icosahedron.
            If False, loads precomputed FreeSurfer icosahedra.
            Default False.

        Returns
        -------
        vertices : np.ndarray, shape (N, 3)
            Array of vertex coordinates.
        triangles : np.ndarray, shape (M, 3)
            Array of triangular faces.

        Examples
        --------
        >>> from nidl.surface.base import Ico
        >>> verts, tris = Ico.mesh(order=3)
        >>> verts.shape
        (642, 3)
        >>> tris.shape
        (1280, 3)
        """
        if standard_ico:
            return cls._generate_standard(order)
        return cls._load_freesurfer(order)

    @classmethod
    @cached
    def patches(
            cls,
            order: int = 3,
            standard_ico: bool = False,
            size: int = 1,
            direct_neighbor: bool = False,
            n_jobs: int = 1,
        ) -> np.ndarray:
        """
        Build triangular patches on an icosahedral mesh.

        This function is used to extract local triangular neighborhoods
        for Vision Transformers.

        Parameters
        ----------
        order : int
            Subdivision order of the icosahedron.
            Default 3.
        standard_ico : bool
            If True, recursively subdivides a standard icosahedron.
            If False, loads precomputed FreeSurfer icosahedra.
            Default False.
        size : int
            Patch subdivision depth.
            Default 1.
        direct_neighbor : bool
            If True, sorts patch vertices by angular order.
            Default False.
        n_jobs : int
            Number of parallel workers.
            Default 1.

        Returns
        -------
        patches : np.ndarray
            For each triangle in the lower-resolution mesh, returns the
            indices of vertices forming its patch.

        Raises
        ------
        ValueError
            If invalid patch size is specified.

        Examples
        --------
        >>> from nidl.surface.base import Ico
        >>> patches = Ico.patches(order=3, size=1)
        >>> patches.shape
        (320, 6)
        """
        # Sanity
        if order - size < 0:
            raise ValueError(
                f"Invalid patch definition: size={size} cannot exceed "
                f"order={order}."
            )

        # High-resolution mesh
        high_verts, _ = cls.mesh(
            order=order,
            standard_ico=standard_ico,
        )

        # Lower-resolution mesh
        low_verts, low_tris = cls.mesh(
            order=order - size,
            standard_ico=standard_ico,
        )

        # Nearest-neighbor mapping
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(high_verts)

        # Build patches in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            patches = list(ex.map(
                functools.partial(
                    cls._patch_iter,
                    vertices=high_verts,
                    lower_vertices=low_verts,
                    size=size,
                    neigh=neigh,
                    direct_neighbor=direct_neighbor,
                ),
                low_tris
            ))

        return np.asarray(patches)

    @classmethod
    @cached
    def neighbors(
            cls,
            order: int = 3,
            standard_ico: bool = False,
            depth: int = 1,
            direct_neighbor: bool = False,
            n_jobs: int = 1,
        ) -> np.ndarray:
        """
        Compute vertex neighborhoods on an icosahedral mesh.

        This function builds multi-ring neighborhoods for each vertex and
        optionally orders direct neighbors according to the DiNe (Direct
        Neighbor) convention.

        Parameters
        ----------
        order : int
            Subdivision order of the icosahedron.
            Default 3.
        standard_ico : bool
            If True, recursively subdivides a standard icosahedron.
            If False, loads precomputed FreeSurfer icosahedra.
            Default False.
        depth : int
            Maximum ring depth. Only neighbors within shortest-path length
            lower or equal to depth are returned.
            Default 1.
        direct_neighbor : bool
            If True, sorts neighbors by angular order following the DiNe
            convention:
            - Regular vertices have 6 neighbors.
            - The 12 special vertices have only 5 neighbors.
            - For 6-neighbor vertices: index 1 is the center, 2-7 are neighbors
              ordered by angle in the tangent plane.
            - For 5-neighbor vertices: indices 1 and 2 both refer to the
              center, and neighbors occupy indices 3-7.
            Default False.
        n_jobs : int
            Number of parallel workers.
            Default 1.

        Returns
        -------
        neighs : np.ndarray
            For each vertex, returns the order or unorder neighbor vertices.

        Examples
        --------
        >>> from nidl.surface.base import Ico
        >>> neighs = Ico.neighbors(order=3, depth=1, direct_neighbor=True)
        >>> neighs[0]
        array([  0,   0, 218, 244, 164, 162, 192])
        """
        # Generate mesh
        verts, tris = cls.mesh(
            order=order,
            standard_ico=standard_ico,
        )

        # Build adjacency graph
        graph = Ico._vertex_adjacency_graph(verts, tris)
        degrees = dict(graph.degree())

        # Process vertices in parallel and sorted order for reproducibility
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            neighs = list(ex.map(
                functools.partial(
                    cls._neighbor_iter,
                    graph=graph,
                    vertices=verts,
                    depth=depth,
                    degrees=degrees,
                    direct_neighbor=direct_neighbor,
                ),
                sorted(graph.nodes)
            ))

        # Create an array: with fixed or varying size elements
        if not direct_neighbor:
            neighs = np.array(neighs, dtype=object)
        else:
            neighs = np.asarray(neighs)

        return neighs

    @staticmethod
    @cached
    def downsample(
            source_order: int,
            target_order: int,
            standard_ico: bool = False,
        ) -> np.ndarray:
        """
        Downsample icosahedron vertices by assigning each target vertex to its
        nearest source vertex.

        Parameters
        ----------
        source_order : int
            Subdivision order of the source icosahedron.
        target_order : int
            Subdivision order of the target icosahedron.
        standard_ico : bool
            If True, recursively subdivides a standard icosahedron.
            If False, loads precomputed FreeSurfer icosahedra.
            Default False.

        Returns
        -------
        indices : np.ndarray, shape (n_target,)
            Indices of the nearest source vertices for each target vertex.

        Examples
        --------
        >>> from nidl.surface import Ico
        >>> indices = Ico.downsample(3, 2)
        >>> print(indices.shape)
        (162,)
        """
        source_verts, _ = Ico.mesh(
            order=source_order,
            standard_ico=standard_ico,
        )
        target_verts, _ = Ico.mesh(
            order=target_order,
            standard_ico=standard_ico,
        )

        if source_verts.size == 0 or target_verts.size == 0:
            return np.empty(0, dtype=int)

        nn = NearestNeighbors(
            n_neighbors=1,
            algorithm="auto",
            leaf_size=2,
        ).fit(source_verts)

        nearest_idx = nn.kneighbors(
            target_verts,
            return_distance=False
        ).squeeze()

        if nearest_idx.size != np.unique(nearest_idx).size:
            raise RuntimeError(
                "Downsampling failed. duplicate nrearest neighbors detected. "
                "Ensure the mesh is a valid icosahedron."
            )

        return nearest_idx


    @staticmethod
    @cached
    def interpolate(
            source_order: int,
            target_order: int,
            standard_ico: bool = False,
        ) -> np.ndarray:
        """
        Interpolate icosahedron missing data by finding nearest neighbors.

        Interpolation weights are set to 1 for a regular icosahedron geometry.

        Parameters
        ----------
        source_order : int
            Subdivision order of the source icosahedron.
        target_order : int
            Subdivision order of the target icosahedron.
        standard_ico : bool
            If True, recursively subdivides a standard icosahedron.
            If False, loads precomputed FreeSurfer icosahedra.
            Default False.

        Returns
        -------
        indices : np.ndarray, shape (n_target, n_feats)
            Interpolation indices.

        Examples
        --------
        >>> from nidl.surface import Ico
        >>> indices = Ico.interpolate(
        ...     source_order=2,
        ...     target_order=3,
        ... )
        >>> indices.shape
        (642, 2)
        """
        # Load mesh
        target_verts, target_tris = Ico.mesh(
            order=target_order,
            standard_ico=standard_ico,
        )

        # Build adjacency graph
        graph = Ico._vertex_adjacency_graph(target_verts, target_tris)

        # Identify vertices that exist in both resolutions
        common_vertices = Ico.downsample(
            target_order, source_order,
            standard_ico=standard_ico,
        )

        # Process vertices in sorted order for reproducibility
        indices = []
        for node in sorted(graph.nodes):
            # Direct mapping
            if node in common_vertices:
                indices.append([node] * 2)

            # Nearest neighbors among common vertices
            else:
                indices.append([
                    idx
                    for idx in graph.neighbors(node)
                    if idx in common_vertices
                ])

        return np.asarray(indices)

    @staticmethod
    def min_rings(n_vertices: int) -> int:
        """
        Compute the minimum k-ring depth required to reach a target number
        of neighboring vertices on an icosahedral mesh.

        Parameters
        ----------
        n_vertices : int
            Desired number of neighboring vertices.

        Returns
        -------
        int
            The minimal k-ring depth needed to include at least `n_vertices`
            neighbors.

        Examples
        --------
        >>> from nidl.surface.base import Ico
        >>> verts, tris = Ico.mesh(order=3)
        >>> Ico.min_rings(len(verts) // 4)
        8
        """
        return int(np.ceil((3 + np.sqrt(12 * n_vertices - 3)) / 6))

    @staticmethod
    def order(n_vertices: int) -> int:
        """
        Compute the subdivision order of an icosahedral mesh from its number
        of vertices.

        The number of vertices for an icosahedron of order k is:
            N = 10 * 4**k + 2

        Parameters
        ----------
        n_vertices : int
            Number of vertices in the mesh.

        Returns
        -------
        int
            The subdivision order k.

        Raises
        ------
        ValueError
            If n_vertices does not correspond to a valid icosahedral mesh.

        Examples
        --------
        >>> from nidl.surface import Ico
        >>> verts, tris = Ico.mesh(order=3)
        >>> Ico.order(len(verts))
        3
        """
        if n_vertices < 12:
            raise ValueError(
                "An icosahedron must have at least 12 vertices."
            )

        # Compute theoretical order
        value = (n_vertices - 2) / 10
        order = np.log(value) / np.log(4)

        # Check if order is (almost) integer
        order_rounded = round(order)
        if not np.isclose(order, order_rounded, atol=1e-9):
            raise ValueError(
                f"{n_vertices} vertices does not match any regular "
                f"icosahedron (expected 10*4^k + 2)."
            )

        return order_rounded

    @staticmethod
    def n_vertices(order: int = 3) -> int:
        """
        Compute the number of vertices of an icosahedron of specific order.

        Parameters
        ----------
        order: int, default 3
            the icosahedron order.

        Returns
        -------
        n_vertices: int
            number of vertices of the corresponding icosahedron

        Examples
        --------
        >>> from nidl.surface import Ico
        >>> Ico.n_vertices(order=3)
        642
        """
        return 10 * 4 ** order + 2

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_standard(
            order: int
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a recursively subdivided standard icosahedron.

        This method starts from the canonical 12-vertex icosahedron and
        performs `order` rounds of triangular subdivision. Each subdivision
        splits every triangle into four smaller triangles and projects new
        vertices onto the unit sphere.

        Parameters
        ----------
        order : int
            Number of recursive subdivisions to apply. An order of 0 returns
            the base icosahedron with 12 vertices and 20 faces.

        Returns
        -------
        vertices : np.ndarray, shape (N, 3)
            Array of vertex coordinates.
        triangles : np.ndarray, shape (M, 3)
            Array of triangular faces.
        """
        R = (1 + np.sqrt(5)) / 2
        vertices = [
            Ico._normalize([-1, R, 0]),
            Ico._normalize([1, R, 0]),
            Ico._normalize([-1, -R, 0]),
            Ico._normalize([1, -R, 0]),
            Ico._normalize([0, -1, R]),
            Ico._normalize([0, 1, R]),
            Ico._normalize([0, -1, -R]),
            Ico._normalize([0, 1, -R]),
            Ico._normalize([R, 0, -1]),
            Ico._normalize([R, 0, 1]),
            Ico._normalize([-R, 0, -1]),
            Ico._normalize([-R, 0, 1]),
        ]
        triangles = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ]
        cache = {}

        for _ in range(order):
            new_tris = []
            for tri in triangles:
                v1 = Ico._midpoint(tri[0], tri[1], vertices, cache)
                v2 = Ico._midpoint(tri[1], tri[2], vertices, cache)
                v3 = Ico._midpoint(tri[2], tri[0], vertices, cache)

                new_tris.extend([
                    [tri[0], v1, v3],
                    [tri[1], v2, v1],
                    [tri[2], v3, v2],
                    [v1, v2, v3],
                ])
            triangles = new_tris

        return np.asarray(vertices), np.asarray(triangles)

    @staticmethod
    def _load_freesurfer(
            order: int,
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load a precomputed FreeSurfer icosahedral mesh.

        FreeSurfer provides a family of spherical meshes named
        ``fsaverage0``, ``fsaverage1``, ..., ``fsaverageN``.
        This method loads the corresponding vertices and triangles from the
        packaged ``freesurfer_icos.npz`` resource file.

        Parameters
        ----------
        order : int
            Mesh resolution level to load. Must correspond to an existing
            ``fsaverage{order}`` entry in the resource file.

        Returns
        -------
        vertices : np.ndarray, shape (N, 3)
            Array of vertex coordinates.
        triangles : np.ndarray, shape (M, 3)
            Array of triangular faces.

        Raises
        ------
        KeyError
            If the requested ``fsaverage{order}`` mesh is not available.
        """
        resource_dir = Path(__file__).parent / "resources"
        resource_file = resource_dir / "freesurfer_icos.npz"
        icos = np.load(resource_file)

        surf_name = f"fsaverage{order}"
        try:
            vertices = icos[f"{surf_name}.vertices"]
            triangles = icos[f"{surf_name}.triangles"]
        except Exception:
            print(f"available topologies: {icos.files}")
            raise

        return vertices, triangles

    @staticmethod
    def _normalize(
            vertex: np.ndarray | list[float],
        ) -> list[float]:
        """
        Normalize a 3D vertex onto the unit sphere.

        Parameters
        ----------
        vertex : np.ndarray | list[float], shape (3, )
            Input 3D coordinates.

        Returns
        -------
        list of float
            Normalized coordinates on the unit sphere.
        """
        x, y, z = vertex
        length = np.sqrt(x**2 + y**2 + z**2)
        return [x / length, y / length, z / length]


    @staticmethod
    def _midpoint(
            i1: int,
            i2: int,
            vertices: list[list[float]],
            cache: dict | None = None,
        ) -> int:
        """
        Compute the midpoint between two vertices, normalize it onto the unit
        sphere, and return its index. Uses a cache to avoid recomputing
        midpoints during icosahedron subdivision.

        Parameters
        ----------
        i1 : int
            Index of the first vertex.
        i2 : int
            Index of the second vertex.
        vertices : list[list[float]], shape (N, 3)
            Mutable list of vertex coordinates. New vertices
            are appended here.
        cache : dict | None
            Mapping of 'i1-i2' to midpoint index to avoid duplicates.

        Returns
        -------
        int
            Index of the midpoint vertex in `vertices`.

        Examples
        --------
        >>> verts = [[0, 0, 1], [0, 1, 0]]
        >>> cache = {}
        >>> idx = Ico._midpoint(0, 1, verts, cache)
        >>> idx
        2
        >>> np.round(verts[idx], 1)
        array([0. , 0.7, 0.7])
        """
        # Ensure consistent ordering for cache key
        i1, i2 = sorted((i1, i2))
        key = f"{i1}-{i2}"

        # Return cached midpoint if available
        if cache is not None and key in cache:
            return cache[key]

        # Compute midpoint
        p1 = vertices[i1]
        p2 = vertices[i2]
        midpoint = [(a + b) / 2.0 for a, b in zip(p1, p2)]

        # Normalize and append
        vertices.append(Ico._normalize(midpoint))
        index = len(vertices) - 1

        # Store in cache
        if cache is not None:
            cache[key] = index

        return index

    @staticmethod
    def _patch_iter(
            tri: np.ndarray,
            vertices: np.ndarray,
            lower_vertices: np.ndarray,
            size: int,
            neigh: NearestNeighbors,
            direct_neighbor: bool
        ) -> list[int]:
        """
        Build a triangular patch from a single triangle.

        Parameters
        ----------
        tri : np.ndarray, shape (3, )
            Triangle indices in the lower-resolution mesh.
        vertices : np.ndarray
            High-resolution vertex coordinates.
        lower_vertices : np.ndarray
            Lower-resolution vertex coordinates.
        size : int
            Subdivision depth.
        neigh : NearestNeighbors
            Fitted nearest-neighbor model.
        direct_neighbor : bool
            Whether to sort patch vertices by angular order.

        Returns
        -------
        locs : list[int]
            Indices of vertices forming the patch.
        """
        # Start from the 3 vertices of the lower-resolution triangle
        _vertices = [lower_vertices[idx] for idx in tri]
        _triangles = [[0, 1, 2]]

        # Subdivide patch
        for _ in range(size):
            subdiv = []
            for t in _triangles:
                v1 = Ico._midpoint(t[0], t[1], _vertices)
                v2 = Ico._midpoint(t[1], t[2], _vertices)
                v3 = Ico._midpoint(t[2], t[0], _vertices)
                subdiv.extend([
                    [t[0], v1, v3],
                    [t[1], v2, v1],
                    [t[2], v3, v2],
                    [v1, v2, v3],
                ])
            _triangles = subdiv

        # Map subdivided vertices back to high-res mesh
        locs = neigh.kneighbors(_vertices, return_distance=False)
        locs = np.unique(locs.squeeze())

        # Optional angular sorting
        if direct_neighbor:
            center = np.mean(lower_vertices[tri], axis=0)
            center /= np.linalg.norm(center)

            angles = np.degrees([
                Ico._get_angle_with_xaxis(center, center, vertices[idx])
                for idx in locs
            ])

            locs = [
                idx
                for _, idx in sorted(zip(angles, locs))
            ]

        return locs

    @staticmethod
    def _neighbor_iter(
            node: int,
            graph: nx.Graph,
            vertices: np.ndarray,
            depth: int,
            degrees: dict,
            direct_neighbor: bool,
        ) -> np.ndarray:
        """
        Build ring neighbors from a node.

        Parameters
        ----------
        node: int
            A node of the graph.
        graph : nx.Graph
            Undirected graph where nodes correspond to vertices and edges
            connect vertices that share a triangle edge.
        vertices : np.ndarray, shape (N, 3)
            Array of vertex coordinates.
        depth : int
            Maximum ring depth. Only neighbors within shortest-path length
            lower or equal to depth are returned.
        degrees: dict
            Mapping nodes to their degree.
        direct_neighbor : bool
            Whether to sort patch vertices by angular order.

        Returns
        -------
        locs : list[int]
            Indices of vertices forming the patch.
        """
        # Collect neighbors by ring (shortest-path length)
        ring_map = {}
        for neigh, dist in nx.single_source_shortest_path_length(
                graph, node, cutoff=depth
            ).items():
            if dist > 0:
                ring_map.setdefault(dist, []).append(neigh)

        # If no angular sorting is requested, store ring structure directly
        if not direct_neighbor:
            return np.concatenate([[node], *list(ring_map.values())])

        # DiNe angular ordering
        ordered = []
        missing = {}  # track 5-neighbor vertices needing duplication
        center_has_missing = False
        total_expected = 0

        for ring, ring_vertices in ring_map.items():

            # Compute angular ordering for this ring
            angles = np.degrees([
                Ico._get_angle_with_xaxis(vertices[node], vertices[node], vec)
                for vec in vertices[ring_vertices]
            ])
            ring_vertices = [
                v for _, v in sorted(zip(angles, ring_vertices))
            ]
            ring_map[ring] = ring_vertices

            # Expected number of neighbors in this ring
            total_expected += 6 * ring

            # Identify 5-neighbor vertices in the previous ring
            prev_ring = ring_map.get(ring - 1, [node] if ring == 1 else [])
            five_neighbor_nodes = [v for v in prev_ring if degrees[v] == 5]

            # Insert previously missing neighbors
            for v, counts in missing.items():
                ring_vertices = [v] * counts[0] + ring_vertices
                missing[v] = counts[1:]

            # Track new missing neighbors
            for v in five_neighbor_nodes:
                missing[v] = list(range(2, depth + 2 - ring))
                if v == node:
                    center_has_missing = True
                else:
                    ordered.insert(ordered.index(v), v)

            ordered.extend(ring_vertices)

        # Insert center vertex (once or twice)
        ordered.insert(0, node)
        if center_has_missing:
            ordered.insert(0, node)

        # Validate DiNe structure
        if len(ordered) != total_expected + 1:
            raise ValueError("Mesh is not an icosahedron.")

        return ordered

    @staticmethod
    def _triangles_to_edges(
            triangles: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert a list of triangular faces into their corresponding edges.

        Parameters
        ----------
        triangles : np.ndarray
            Integer vertex indices defining each triangle with shape
            (M, 3).

        Returns
        -------
        edges : np.ndarray, shape (M * 3, 2)
            Array of vertex index pairs representing the three edges of each
            triangle. Edges are not deduplicated, each
            triangle contributes exactly three edges.
        triangles_index : np.ndarray, shape (M * 3, )
            For each edge in `edges`, gives the index of the triangle it came
            from.

        Notes
        -----
        For a triangle (a, b, c), the edges returned are:
        (a, b), (b, c), (c, a)

        Examples
        --------
        >>> triangles = np.array([[0, 1, 2],
        ...                       [2, 3, 4]])
        >>> edges, idx = Ico._triangles_to_edges(triangles)
        >>> edges
        array([[0, 1],
               [1, 2],
               [2, 0],
               [2, 3],
               [3, 4],
               [4, 2]])
        >>> idx
        array([0, 0, 0, 1, 1, 1])
        """
        # Each triangle contributes three edges
        edges = triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)

        # For each triangle, repeat its index 3 times (one per edge)
        triangles_index = np.repeat(np.arange(len(triangles)), 3)

        return edges, triangles_index

    @staticmethod
    def _vertex_adjacency_graph(
            vertices: np.ndarray,
            triangles: np.ndarray,
        ) -> nx.Graph:
        """
        Construct a NetworkX graph where each vertex in the mesh is a node
        and edges represent adjacency relationships derived from triangle
        faces.

        Parameters
        ----------
        vertices : np.ndarray, shape (N, 3)
            Array of vertex coordinates.
        triangles : np.ndarray, shape (M, 3)
            Array of triangular faces, each row containing three vertex
            indices.

        Returns
        -------
        graph : networkx.Graph
            Undirected graph where nodes correspond to vertices and edges
            connect vertices that share a triangle edge.

        Examples
        --------
        >>> from nidl.surface import Ico
        >>> verts, tris = Ico.mesh(order=2)
        >>> graph = Ico._vertex_adjacency_graph(verts, tris)
        >>> list(graph.neighbors(0))
        [65, 58, 44, 42, 51]
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(vertices)))

        # Convert triangles to unique edges
        edges, _ = Ico._triangles_to_edges(triangles)

        # Add edges directly, NetworkX automatically avoids duplicates
        for i, j in edges:
            graph.add_edge(min(i, j), max(i, j))

        return graph

    @staticmethod
    def _get_angle_with_xaxis(
            center: np.ndarray,
            normal: np.ndarray,
            point: np.ndarray,
        ) -> float:
        """
        Project a point onto the tangent plane of a sphere and compute its
        angle relative to the x-axis in that plane.

        Parameters
        ----------
        center : np.ndarray, shape (3,)
            Point on the sphere surface (projection center).
        normal : np.ndarray, shape (3,)
            Normal vector of the tangent plane.
        point : np.ndarray, shape (3,)
            Point to project.

        Returns
        -------
        angle : float
            Angle in radians between the projected point and the x-axis
            of the tangent plane.
        """
        center = np.asarray(center)
        normal = np.asarray(normal)
        point = np.asarray(point)

        # Project point onto tangent plane
        vec = point - center
        projection = point - normal * np.dot(vec, normal)

        # Construct tangent-plane axes (nx, ny)
        if center[0] != 0 or center[1] != 0:
            nx = np.cross([0, 0, 1], center)
            ny = np.cross(center, nx)
        else:
            nx = np.array([1, 0, 0])
            ny = np.array([0, 1, 0])

        # Normalize vectors
        vec = projection - center
        if np.linalg.norm(vec) != 0:
            vec = vec / np.linalg.norm(vec)

        nx = nx / np.linalg.norm(nx)
        ny = ny / np.linalg.norm(ny)

        # Compute angle using arccos of dot product
        cos_theta = np.clip(np.dot(vec, nx), -1.0, 1.0)
        angle = np.arccos(cos_theta)

        # Adjust orientation using ny
        if np.dot(vec, ny) < 0:
            angle = 2 * np.pi - angle

        return angle
