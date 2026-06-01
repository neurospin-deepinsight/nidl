##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Spherical layers.
"""

import collections

import numpy as np
import torch
from torch import nn

from .. import Ico


class IcoConv(nn.Module):
    """
    Direct Neighbor (DiNe) convolutional layer for signals defined on an
    icosahedrally discretized sphere, using an n-ring neighborhood filter.

    Parameters
    ----------
    in_feats : int
        Number of input feature channels.
    out_feats : int
        Number of output feature channels.
    order : int
        Subdivision order of the icosahedron mesh.
    depth : int
        Maximum geodesic ring depth. Only neighbors whose shortest-path
        distance is less than or equal to `depth` are included.
    standard_ico : bool
        If True, recursively subdivides a standard icosahedron.
        If False, loads precomputed FreeSurfer icosahedral meshes.
        Default False.
    bias : bool
        If True, adds a learnable bias term.
        Default True.

    Examples
    --------
    >>> import torch
    >>> from nidl.surface import Ico
    >>> from nidl.surface.nn import IcoConv
    >>> module = IcoConv(
    ...     in_feats=8,
    ...     out_feats=8,
    ...     order=2,
    ...     depth=1,
    ... )
    >>> x = torch.zeros((10, 8, Ico.n_vertices(order=2)))
    >>> y = module(x)
    >>> y.shape
    torch.Size([10, 8, 162])
    """
    def __init__(
            self,
            in_feats: int,
            out_feats: int,
            order: int,
            depth: int,
            standard_ico: bool = False,
            bias: bool = True,
        ) -> None:
        super().__init__()

        # Precompute neighbors
        self.neighbors = Ico.neighbors(
            order=order,
            standard_ico=standard_ico,
            depth=depth,
            direct_neighbor=True,
        )
        self.n_vertices, self.n_neighbors = self.neighbors.shape

        # Linear kernel applied to flattened neighborhood features
        self.weight = nn.Linear(
            in_feats * self.n_neighbors,
            out_feats,
            bias=bias,
        )

        self.in_feats = in_feats
        self.out_feats = out_feats

    def forward(
            self,
            x: torch.Tensor,
        ) -> torch.Tensor:
        """
        Forward method.

        Parameters
        ----------
        x: torch.Tensor, shape (S, C, N)
            Input texture, where S is the number of samples, C the number of
            input features/channels, and N the number of icosahedron vertices.

        Returns
        -------
        torch.Tensor, shape (S, out_feats, N)
            Output after DiNe convolution.
        """
        S, C, N = x.shape
        assert self.in_feats == C, "Channel mismatch."
        assert self.n_vertices == N, "Vertex count mismatch."

        # Gather neighbors: (S*N, C*K)
        x = x[:, :, self.neighbors.reshape(-1)]
        x = x.view(S, self.in_feats, self.n_vertices, self.n_neighbors)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(
            len(x) * self.n_vertices,
            self.in_feats * self.n_neighbors
        )

        # Apply linear kernel
        out = self.weight(x)

        # Reshape back to (S, out_feats, N)
        out = out.view(
            S,
            self.n_vertices,
            self.out_feats
        )
        out = out.permute(0, 2, 1)

        return out


class IcoPool(nn.Module):
    """
    Pooling layer for signals defined on an icosahedrally discretized sphere,
    using a 1-ring neighborhood filter.

    Parameters
    ----------
    order : int
        Subdivision order of the icosahedron mesh.
    standard_ico : bool
        If True, recursively subdivides a standard icosahedron.
        If False, loads precomputed FreeSurfer icosahedral meshes.
        Default False.
    pooling_type : str
        Pooling operation to apply. Supported options are 'mean' and 'max'.
        Default 'mean'.

    Raises
    ------
    ValueError
        If the input pooling type is not supported.

    Examples
    --------
    >>> import torch
    >>> from nidl.surface import Ico
    >>> from nidl.surface.nn import IcoPool
    >>> module = IcoPool(
    ...     order=3,
    ...     pooling_type="mean",
    ... )
    >>> x = torch.zeros((10, 4, Ico.n_vertices(order=3)))
    >>> y = module(x)
    >>> y.shape
    torch.Size([10, 4, 162])
    """
    def __init__(
            self,
            order: int,
            standard_ico: bool = False,
            pooling_type: str = "mean",
        ) -> None:
        super().__init__()

        # Precompute downsampling indices
        down_indices = Ico.downsample(
            source_order=order,
            target_order=order - 1,
        )

        # Precompute 1-ring neighbors for the lower-resolution mesh
        neigh = Ico.neighbors(
            order=order - 1,
            standard_ico=standard_ico,
            depth=1,
            direct_neighbor=True,
        )

        # Select only the neighbors corresponding to downsampled vertices
        self.down_neighbors = neigh[down_indices]
        self.n_vertices, self.n_neighbors = self.down_neighbors.shape

        if pooling_type not in ("mean", "max"):
            raise ValueError("Pooling type must be 'mean' or 'max'.")
        self.pooling_type = pooling_type

    def forward(
            self,
            x: torch.Tensor,
        ) -> torch.Tensor:
        """
        Forward method.

        Parameters
        ----------
        x: torch.Tensor, shape (S, C, N)
            Input texture, where S is the batch size, C is channels, and N
            is number of vertices at the high-resolution level.

        Returns
        -------
        torch.Tensor, shape (S, C, M)
            Pooled signal, where M is number of vertices of the
            lower-resolution level.
        """
        # Infer expected number of vertices from input
        expected_n_vertices = int((x.size(2) + 6) / 4)
        assert self.n_vertices == expected_n_vertices, (
            f"Input has {x.size(2)} vertices, but pooling expects "
            f"{4 * self.n_vertices - 6}."
        )

        # Gather neighbors: (S, C, M, K)
        x = x[:, :, self.down_neighbors.reshape(-1)]
        x = x.view(x.size(0), x.size(1), self.n_vertices, self.n_neighbors)

        # Apply pooling
        if self.pooling_type == "mean":
            return torch.mean(x, dim=-1)
        else:
            return torch.max(x, dim=-1).values


class IcoUpConv(nn.Module):
    """
    Transposed convolution layer on an icosahedrally discretized sphere
    using a 1-ring neighborhood filter.

    Parameters
    ----------
    in_feats : int
        Number of input feature channels.
    out_feats : int
        Number of output feature channels.
    order : int
        Subdivision order of the icosahedron mesh.
    standard_ico : bool
        If True, recursively subdivides a standard icosahedron.
        If False, loads precomputed FreeSurfer icosahedral meshes.
        Default False.

    Examples
    --------
    >>> import torch
    >>> from nidl.surface import Ico
    >>> from nidl.surface.nn import IcoUpConv
    >>> module = IcoUpConv(
    ...     in_feats=8,
    ...     out_feats=4,
    ...     order=2,
    ... )
    >>> x = torch.zeros((10, 8, Ico.n_vertices(order=2)))
    >>> y = module(x)
    >>> y.shape
    torch.Size([10, 4, 642])
    """
    def __init__(
            self,
            in_feats: int,
            out_feats: int,
            order: int,
            standard_ico: bool = False,
        ) -> None:
        super().__init__()

        # Precompute downsampling indices
        down_indices = Ico.downsample(
            source_order=order + 1,
            target_order=order,
            standard_ico=standard_ico,
        )

        # Precompute 1-ring neighbors for the lower-resolution mesh
        up_neighbors = Ico.neighbors(
            order=order + 1,
            standard_ico=standard_ico,
            depth=1,
            direct_neighbor=True,
        )
        self.n_up_vertices, self.n_up_neighbors = up_neighbors.shape

        # Select only the neighbors corresponding to dowsampled vertices
        self.neighbors = up_neighbors[down_indices]
        self.n_vertices, self.n_neighbors = self.neighbors.shape

        # Flattened neighbor list for sorting
        flat_neigbors = self.neighbors.reshape(-1)
        argsort = np.argsort(flat_neigbors)
        sorted_flat_neigbors = flat_neigbors[argsort]

        # Ensure coverage of all high-res vertices
        assert np.array_equal(
            np.unique(sorted_flat_neigbors),
            np.arange(self.n_up_vertices)
        )

        # Split sorted neighborhood indices
        _idx1 = 24
        _idx2 = len(down_indices) + 12
        self.sorted_2occ_12neighbors = sorted_flat_neigbors[:_idx1]
        self.sorted_1occ_neighbors = sorted_flat_neigbors[_idx1:_idx2]
        self.sorted_2occ_neighbors = sorted_flat_neigbors[_idx2:]

        # Split argsort neighborhood indices
        self.argsort_2occ_12neighbors = argsort[:_idx1]
        self.argsort_1occ_neighbors = argsort[_idx1:_idx2]
        self.argsort_2occ_neighbors = argsort[_idx2:]

        # Validate occurrences
        self._check_occurence(self.sorted_2occ_12neighbors, occ=2)
        self._check_occurence(self.sorted_1occ_neighbors, occ=1)
        self._check_occurence(self.sorted_2occ_neighbors, occ=2)

        # Linear kernel applied per low-res vertex
        self.weight = nn.Linear(in_feats, out_feats * self.n_up_neighbors)

        self.in_feats = in_feats
        self.out_feats = out_feats

    def _check_occurence(
            self,
            data: np.ndarray,
            occ: int,
        ) -> None:
        """
        Check the occurrence of each element in the data array.

        Parameters
        ----------
        data : np.ndarray
            Input data array.
        occ : int
            Expected occurrence count.

        Raises
        ------
        AssertionError
            If the unique count of occurrences is not equal to 1 or the
            unique count is not equal to the expected occurrence count.
        """
        count = collections.Counter(data)
        unique_count = np.unique(list(count.values()))
        if len(unique_count) != 1 or unique_count[0] != occ:
            raise AssertionError(
                f"Expected occurrence count {occ}, but got {unique_count}"
            )

    def forward(
            self,
            x: torch.Tensor,
        ) -> torch.Tensor:
        """
        Forward method.

        Parameters
        ----------
        x: torch.Tensor, shape (S, C, N)
            Input texture, where S is the batch size, C is channels, and N
            is number of vertices at the low-resolution level.

        Returns
        -------
        torch.Tensor, shape (S, out_feats, M)
            Output after DiNe transpose convolution, where M is number of
            vertices of the higher-resolution level.
        """
        S, C, N = x.shape
        assert self.in_feats == C, "Channel mismatch."
        assert self.n_vertices == N, "Vertex count mismatch."

        # Gather data: (S * N, C)
        x = x.permute(0, 2, 1)
        x = x.reshape(S * N, C)

        # Apply linear kernel per vertex
        x = self.weight(x)

        # Flatten neighbors: (S, N*K, out_feats)
        x = x.view(S, N, self.n_up_neighbors, self.out_feats)
        x = x.view(S, N * self.n_up_neighbors, self.out_feats)

        # Split by occurrence count
        x1 = x[:, self.argsort_2occ_12neighbors].view(
            S, -1, 2, self.out_feats
        )
        x2 = x[:, self.argsort_1occ_neighbors]
        x3 = x[:, self.argsort_2occ_neighbors].view(
            S, -1, 2, self.out_feats
        )

        # Concatenate back
        x = torch.cat(
            (torch.mean(x1, dim=2), x2, torch.mean(x3, dim=2)),
            dim=1,
        )

        # Final shape (S, out_feats, M)
        return x.permute(0, 2, 1)


class IcoUpSample(nn.Module):
    """
    Up sampling layer on an icosahedrally discretized sphere
    using one of mean, max or zero padding sampling strategy.

    This module increases the mesh resolution from order `k` to `k+1`
    by assigning each high-resolution vertex the average/max/padded of its
    corresponding low-resolution neighbors, followed by a linear
    projection of feature channels.

    Parameters
    ----------
    in_feats : int
        Number of input feature channels.
    out_feats : int
        Number of output feature channels.
    order : int
        Subdivision order of the icosahedron mesh.
    standard_ico : bool
        If True, recursively subdivides a standard icosahedron.
        If False, loads precomputed FreeSurfer icosahedral meshes.
        Default False.
    sampling_type : str
        Sampling operation to apply. Supported options are 'mean', 'max' and
        'zeropad'.
        Default 'mean'

    Raises
    ------
    ValueError
        If the input sampling type is not supported.

    Examples
    --------
    >>> import torch
    >>> from nidl.surface import Ico
    >>> from nidl.surface.nn import IcoUpSample
    >>> module = IcoUpSample(
    ...     in_feats=8,
    ...     out_feats=4,
    ...     order=2,
    ... )
    >>> x = torch.zeros((10, 8, Ico.n_vertices(order=2)))
    >>> y = module(x)
    >>> y.shape
    torch.Size([10, 4, 642])
    """
    def __init__(
            self,
            in_feats: int,
            out_feats: int,
            order: int,
            standard_ico: bool = False,
            sampling_type: str = "mean",
        ) -> None:
        super().__init__()

        # Precompute interpolation indices
        self.up_neighbors = Ico.interpolate(
            source_order=order,
            target_order=order + 1,
            standard_ico=standard_ico,
        )
        self.n_vertices, self.n_neighbors = self.up_neighbors.shape

        # Precompute zero-padding mask
        # Zero-pad only vertices whose neighbor list is not a single repeated
        # index
        self.zero_indices = np.where(
            np.array([len(np.unique(row)) > 1 for row in self.up_neighbors])
        )[0]

        # Linear projection of feature channels
        self.fc = nn.Linear(in_feats, out_feats)

        # Validate sampling type
        if sampling_type not in ("mean", "max", "zeropad"):
            raise ValueError(
                "Sampling type must be 'mean', 'max', or 'zeropad'."
            )
        self.sampling_type = sampling_type

        self.in_feats = in_feats
        self.out_feats = out_feats

    def forward(
            self,
            x: torch.Tensor,
        ) -> torch.Tensor:
        """
        Forward method.

        Parameters
        ----------
        x: torch.Tensor, shape (S, C, N)
            Input texture, where S is the batch size, C is channels, and N
            is number of vertices at the low-resolution level.

        Returns
        -------
        torch.Tensor, shape (S, out_feats, M)
            Upsamples features, where M is number of vertices of the
            higher-resolution level.
        """
        S, C, N = x.shape
        assert self.in_feats == C, "Channel mismatch."
        assert self.n_vertices == N * 4 - 6, "Vertex count mismatch."

        if self.sampling_type in ("mean", "max"):
            # Gather neighbors: (S, C, M, K)
            x = x[:, :, self.up_neighbors.reshape(-1)]
            x = x.view(S, C, self.n_vertices, self.n_neighbors)

            # Average over neighbors: (S, C, M)
            if self.sampling_type == "mean":
                x = torch.mean(x, dim=-1)
            else:
                x = x.max(dim=-1).values

        else:
            # Zero-padding
            # Use only the first neighbor: (S, C, M)
            x = x[:, :, self.up_neighbors[:, 0]]

            # Zero out vertices that require padding
            x[:, :, self.zero_indices] = 0

        # Linear projection: reshape to (S * M, C)
        x = x.permute(0, 2, 1).reshape(S * self.n_vertices, self.in_feats)
        x = self.fc(x)

        # Back to (S, out_feats, M)
        x = x.view(S, self.n_vertices, self.out_feats)
        return x.permute(0, 2, 1)
