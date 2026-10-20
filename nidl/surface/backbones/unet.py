##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Spherical UNet architecture.
"""

from pathlib import Path
from typing import Callable, Self

import torch
from torch import nn

from .. import Ico
from ..nn import (
    IcoConv,
    IcoPool,
    IcoUpConv,
    IcoUpSample,
)


class SphericalUNet(nn.Module):
    """
    Spherical U-Net architecture [1]_.

    The architecture is built upon specific spherical surface convolution,
    pooling, and transposed convolution modules. It has an encoder path and
    a decoder path, with a user-defined resolution steps. Different from the
    standard U-Net, all 3x3 convolution are replaced with the DiNe
    convolution, 2x2 up-convolution with surface transposed convolution or
    surface upsampling, and 2x2 max pooling with surface max/mean pooling.
    In addition to the standard U-Net, before each convolution layer's
    rectified linear units (ReLU) activation function, a batch normalization
    layer is added. At the final layer, 1x1 convolution is replaced by
    vertex-wise filter. The number of feature channels are double after each
    surface pooling layer and halve at each transposed convolution or up
    sampling layer.

    Parameters
    ----------
    order : int
        Subdivision order of the icosahedron.
        Default 5.
    standard_ico : bool
        If True, recursively subdivides a standard icosahedron.
        If False, loads precomputed FreeSurfer icosahedra.
        Default False.
    depth : int
        Maximum ring depth. Only neighbors within shortest-path length
        lower or equal to depth are returned.
        Default 1.
    in_channels : int
        Number of input channels.
        Default 1.
    out_channels : int
        Number of output channels.
        Default 1.
    start_filts : int
        Number of convolutional filters for the first conv.
        Default 32.
    n_layers : int
        Number of layers in the architecture.
        Default 3.
    up_mode : str
        Upsampling mode:
        - 'transpose': Transpose convolution (1 ring)
        - 'mean': Nearest neighbor linear interpolation
        - 'max': Max pooling shifted zero padding
        - 'zeropad': Classical zero padding
        Default 'mean'.
    cache_file : str | Path | None
        Path to a a ``.npz``  file used to save and restore precomputed Ico
        data. If ``None``, no caching initilization is performed and data are
        recomputed on initialization.
    printer : Callable | None
        A callable used to output messages. It must accept a single string
        argument. When set, all messages passed to ``print()`` are forwarded
        to this callable. If ``None``, message printing is disabled.
        Default None.
    n_jobs : int
        Number of parallel workers for Ico topology information initialization.
        Default 1.

    Examples
    --------
    >>> from nidl.surface.backbones import SphericalUNet
    >>> model = SphericalUNet(
    ...     order=3,
    ...     depth=1,
    ...     in_channels=2,
    ...     out_channels=4,
    ...     start_filts=8,
    ...     n_layers=2,
    ...     up_mode="mean",
    ...     standard_ico=False,
    )
    >>> print(model)

    References
    ----------
    .. [1] Zhao F, et al., Spherical U-Net on Cortical Surfaces: Methods and
       Applications, IPMI, 2019.
    """
    def __init__(
            self,
            order: int = 5,
            standard_ico: bool = False,
            depth: int = 1,
            in_channels: int = 1,
            out_channels: int = 1,
            start_filts: int = 32,
            n_layers: int = 3,
            up_mode: str = "mean",
            cache_file: str | Path | None = None,
            printer: Callable | None = None,
            n_jobs: int = 1,
        ) -> None:
        super().__init__()

        self.order = order
        self.standard_ico = standard_ico
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_filts = start_filts
        self.n_layers = n_layers
        self.up_mode = up_mode
        self.printer = printer

        if not (1 <= depth < order):
            raise ValueError(f"Invalid depth: {depth}")

        self.n_vertices = Ico.n_vertices(order=order)
        self.filts = [
            in_channels,
            *[
                start_filts * 2 ** idx
                for idx in range(n_layers)
            ],
        ]

        # Branches
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        # Build blocks
        self.print(
            "SphericalUNet initialization may take some time if no caching "
            "initialization is used or the cache is empty."
        )
        kwargs = {
            "standard_ico": standard_ico,
            "depth": depth,
            "n_jobs" : n_jobs,
        }
        with Ico.cachemanager(cache_file):
            for level in range(1, n_layers + 1):
                self.print(
                    f"- [SphericalUNet] encoder initilization step: {level} / "
                    f"{n_layers}"
                )
                block_order = order - level + 1

                self.encoder.add_module(
                    f"down{level}",
                    self._down_block(
                        in_ch=self.filts[level - 1],
                        out_ch=self.filts[level],
                        order=block_order,
                        pool_mode=(
                            None if level == 1
                            else "max" if up_mode == "max"
                            else "mean"
                        ),
                        **kwargs
                    )
                )

            for cnt, level in enumerate(range(n_layers , 1, -1), start=1):
                self.print(
                    "- [SphericalUNet] decoder initilization step: "
                    f"{cnt} / {n_layers - 1}"
                )
                block_order = order - level + 1

                self.decoder.add_module(
                    f"up{cnt}",
                    self._up_block(
                        in_ch=self.filts[level],
                        out_ch=self.filts[level - 1],
                        order=block_order,
                        up_mode=up_mode,
                        **kwargs
                    )
                )

            self.fc = nn.Linear(self.filts[1], out_channels)

    def forward(
            self,
            x: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, n_channels, n_vertices)``.
            It may contain either left- or right-hemisphere data, but must
            first be symmetrized (as in an xhemi-style dataset).

        Returns
        -------
        torch.Tensor
            Decoded input of shape ``(batch_size, n_channels, n_vertices)``.

        Raises
        ------
        ValueError
            If the input data has incompatible shape.
        """
        if x.size(2) != self.n_vertices:
            raise ValueError(
                f"Input data must be projected on an {self.order} order "
                "icosahedron."
            )

        skip_connections = []
        for level in range(1, self.n_layers + 1):
            down_block = getattr(self.encoder, f"down{level}")
            x = down_block(x)
            if level < self.n_layers:
                skip_connections.append(x)
        for level in range(1, self.n_layers):
            up_block = getattr(self.decoder, f"up{level}")
            x = up_block.up(x)
            x_skip = skip_connections.pop()
            x = torch.cat((x, x_skip), 1)
            x = up_block.double_conv(x)

        n_samples = len(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(n_samples * self.n_vertices, self.filts[1])
        x = self.fc(x)
        x = x.view(n_samples, self.n_vertices, self.out_channels)
        x = x.permute(0, 2, 1)

        return x

    def print(
            self,
            message: str
        ) -> None:
        """
        Print a message using the configured printer.

        Parameters
        ----------
        message : str
            The text message to output. If a printer function is set on the
            instance, the message is forwarded to that function.

        Notes
        -----
        This method does nothing when ``self.printer`` is ``None``.
        """
        if self.printer is not None:
            self.printer(message)

    @classmethod
    def from_pretrained(
            cls,
            name: str,
            weights_file: str | Path | None = None,
            cache_file: str | Path | None = None,
            printer: Callable | None = None,
            n_jobs: int = 1,
        ) -> Self:
        """
        Create a model instance and load pretrained weights.

        Parameters
        ----------
        name : str
            Name of of model to be loaded.
        weights_file : str | Path | None
            Path to a .pt file containing a state_dict. If None the file
            is fetched from hugging face.
        cache_file : str | Path | None
            Path to a a ``.npz``  file used to save and restore precomputed
            Ico data. If ``None``, no caching initilization is performed and
            data are recomputed on initialization.
        printer : Callable | None
            A callable used to output messages. It must accept a single string
            argument. When set, all messages passed to ``print()`` are
            forwarded to this callable. If ``None``, message printing is
            disabled.
            Default None.
        n_jobs : int
            Number of parallel workers for Ico topology inforamtion
            initialization.
            Default 1.

        Returns
        -------
        Self
            Model with loaded weights.

        Raises
        ------
        ValueError
            If the pretrained model or model weights in not recognize.
        """
        raise NotImplementedError

    @classmethod
    def _down_block(
            cls,
            standard_ico: bool,
            depth: int,
            in_ch: int,
            out_ch: int,
            order: int,
            pool_mode: str | None,
            n_jobs: int,
        ) -> nn.Sequential:
        """
        Create a single SphericalUNet downsamling block as a sequential
        container.

        Each block consists of repeating two times:
        - a DiNe convolution (`IcoConv`)
        - a batch normalization (`BatchNorm1d`)
        - an activation function (`LeakyReLU`)

        Optionally a pooling is added first.

        Parameters
        ----------
        standard_ico : bool
            If True, recursively subdivides a standard icosahedron.
            If False, loads precomputed FreeSurfer icosahedra.
        depth : int
            Maximum ring depth. Only neighbors within shortest-path length
            lower or equal to depth are returned.
        in_ch : int
            Number of input feature channels.
        out_ch : int
            Number of output feature channels.
        order : int
            Icosahedron subdivision order for the convolution.
        pool_mode : str | None
            Pooling mode: 'mean' or 'max'. If ``None``, no polling is
            performed.
        n_jobs : int
            Number of parallel workers.

        Returns
        -------
        seq : nn.Sequential
            The module container to which the block is appended.
        """
        seq = nn.Sequential()
        if pool_mode is not None:
            seq.add_module(
                "pool",
                IcoPool(
                    order=order + 1,
                    standard_ico=standard_ico,
                    pooling_type=pool_mode,
                    n_jobs=n_jobs,
                )
            )
        seq.add_module(
            "double_conv",
            cls._double_conv(
                standard_ico,
                depth,
                in_ch,
                out_ch,
                order,
                n_jobs,
            )
        )
        return seq

    @classmethod
    def _up_block(
            cls,
            standard_ico: bool,
            depth: int,
            in_ch: int,
            out_ch: int,
            order: int,
            up_mode: str,
            n_jobs: int,
        ) -> nn.Sequential:
        """
        Create a single SphericalUNet upsampling block as a sequential
        container.

        Each block consists of repeating two times:
        - a DiNe convolution (`IcoConv`)
        - a batch normalization (`BatchNorm1d`)
        - an activation function (`LeakyReLU`)

        The upsampling is added first.

        Parameters
        ----------
        standard_ico : bool
            If True, recursively subdivides a standard icosahedron.
            If False, loads precomputed FreeSurfer icosahedra.
        depth : int
            Maximum ring depth. Only neighbors within shortest-path length
            lower or equal to depth are returned.
        in_ch : int
            Number of input feature channels.
        out_ch : int
            Number of output feature channels.
        order : int
            Icosahedron subdivision order for the convolution.
        up_mode : str
            Upsampling mode:
            - 'transpose': Transpose convolution (1 ring)
            - 'mean': Nearest neighbor linear interpolation
            - 'max': Max pooling shifted zero padding
            - 'zeropad': Classical zero padding
        n_jobs : int
            Number of parallel workers.

        Returns
        -------
        seq : nn.Sequential
            The module container to which the block is appended.

        Raises
        ------
        ValueError
            If the upsampling mode is not supported.
        """
        seq = nn.Sequential()
        if up_mode in ("mean", "max", "zeropad"):
            upsampling = IcoUpSample(
                in_ch,
                out_ch,
                order=order,
                standard_ico=standard_ico,
                sampling_type=up_mode,
            )
        elif up_mode == "transpose":
            upsampling = IcoUpConv(
                in_ch,
                out_ch,
                order=order,
                standard_ico=standard_ico,
                n_jobs=n_jobs,
            )
        else:
            raise ValueError("Upsampling mode is not supported")
        seq.add_module(
            "up",
            upsampling
        )
        seq.add_module(
            "double_conv",
            cls._double_conv(
                standard_ico,
                depth,
                in_ch,
                out_ch,
                order + 1,
                n_jobs,
            )
        )
        return seq

    @classmethod
    def _double_conv(
            cls,
            standard_ico: bool,
            depth: int,
            in_ch: int,
            out_ch: int,
            order: int,
            n_jobs: int,
        ) -> nn.Sequential:
        """
        Create a single SphericalUNet double convolution block as a
        sequential container.

        Each block consists of repeating two times:
        - a DiNe convolution (`IcoConv`)
        - a batch normalization (`BatchNorm1d`)
        - an activation function (`LeakyReLU`)

        Parameters
        ----------
        standard_ico : bool
            If True, recursively subdivides a standard icosahedron.
            If False, loads precomputed FreeSurfer icosahedra.
        depth : int
            Maximum ring depth. Only neighbors within shortest-path length
            lower or equal to depth are returned.
        in_ch : int
            Number of input feature channels.
        out_ch : int
            Number of output feature channels.
        order : int
            Icosahedron subdivision order for the convolution.
        pool_mode : str | None
            Pooling mode: 'mean' or 'max'. If ``None``, no polling is
            performed.
        n_jobs : int
            Number of parallel workers.

        Returns
        -------
        seq : nn.Sequential
            The module container to which the block is appended.
        """
        seq = nn.Sequential()
        for idx in range(1, 3):
            seq.add_module(
                f"conv_{idx}",
                IcoConv(
                    in_ch if idx == 1 else out_ch,
                    out_ch,
                    order=order,
                    depth=depth,
                    standard_ico=standard_ico,
                    n_jobs=n_jobs,
                )
            )
            seq.add_module(
                f"bn__{idx}",
                nn.BatchNorm1d(
                    out_ch,
                    momentum=0.15,
                    affine=True,
                    track_running_stats=False,
                )
            )
            seq.add_module(
                f"act_{idx}",
                nn.LeakyReLU(0.2, inplace=True)
            )
        return seq
