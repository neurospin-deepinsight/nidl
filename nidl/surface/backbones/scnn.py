##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Spherical convolutional neural networks.
"""

from pathlib import Path
from typing import Callable, Self

import torch
from torch import nn

from .. import Ico
from ..nn import IcoConv, IcoPool


class HemiSCNN(nn.Module):
    """
    Spherical convolutional encoder with dual-hemisphere processing and
    mid-level early feature fusion [1]_.

    This model implements a four-block spherical CNN (SCNN) encoder. Each block
    consists of a DiNe convolution, an activation function, and an average-pool
    operator. The first `fusion_level` blocks process the left and right
    hemispheres independently; their feature maps are then concatenated along
    the channel dimension. All subsequent blocks operate on the fused
    representation. The final output is projected into a latent embedding.

    Parameters
    ----------
    order : int
        Subdivision order of the icosahedron.
        Default 7.
    standard_ico : bool
        If True, recursively subdivides a standard icosahedron.
        If False, loads precomputed FreeSurfer icosahedra.
        Default False.
    depth : int
        Maximum ring depth. Only neighbors within shortest-path length
        lower or equal to depth are returned.
        Default 1.
    input_channels : int
        Number of input channels per hemisphere.
        Default 1.
    latent_dim : int
        Dimensionality of the latent representation produced by the encoder.
        Default 128.
    convs : tuple[int]
        Number of convolutional filters in each block.
        Default (128, 128, 256, 256)
    fusion_level : int,
        Block index at which left/right hemisphere features are concatenated.
        Default 1.
    activation : str
        Name of the activation class from `torch.nn` to apply after each
        convolution layer.
        Default 'ReLU'
    batch_norm : bool
        If True, applies batch normalization after each convolution.
        Default True.
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
    >>> from nidl.surface.backbones import HemiSCNN
    >>> model = HemiSCNN(
    ...     order=3,
    ...     convs=[128, 256],
    ...     printer=print,
    )
    >>> print(model)

    References
    ----------
    .. [1] Corentin Ambroise et al., "MixUp brain-cortical augmentations in
       self-supervised learning", MLCN 2023.
    """
    def __init__(
            self,
            order: int = 5,
            standard_ico: bool = False,
            depth: int = 1,
            input_channels: int = 1,
            latent_dim: int = 128,
            convs: tuple[int] = (128, 128, 256, 256),
            fusion_level: int = 1,
            activation: str = "ReLU",
            batch_norm: bool = True,
            cache_file: str | Path | None = None,
            printer: Callable | None = None,
            n_jobs: int = 1,
        ) -> None:
        super().__init__()

        self.order = order
        self.standard_ico = standard_ico
        self.depth = depth
        self.latent_dim = latent_dim
        self.fusion_level = fusion_level
        self.printer = printer

        if not (1 <= fusion_level <= len(convs)):
            raise ValueError(f"Invalid fusion level: {fusion_level}")

        # Branches
        self.left_conv = nn.Sequential()
        self.right_conv = nn.Sequential()
        self.fused_conv = nn.Sequential()

        # Build blocks
        self.printer(
            "HemiSCNN initialization may take some time if no caching "
            "initialization is used or the cache is empty."
        )
        with Ico.cachemanager(cache_file):
            in_ch = input_channels
            for level, out_ch in enumerate(convs):
                level += 1
                self.printer(
                    f"- [HemiSCNN] initilization step: {level} / {len(convs)}"
                )
                block_order = order - level + 1

                kwargs = {
                    "level": level,
                    "standard_ico": standard_ico,
                    "depth": depth,
                    "order": block_order,
                    "batch_norm": batch_norm,
                    "activation": activation,
                    "n_jobs" : n_jobs,
                }

                # Before fusion: separate left/right branches
                if level <= fusion_level:
                    half_out_ch = out_ch // 2
                    self._conv_block(
                        seq=self.left_conv,
                        in_ch=in_ch,
                        out_ch=half_out_ch,
                        **kwargs
                    )
                    self._conv_block(
                        seq=self.right_conv,
                        in_ch=in_ch,
                        out_ch=half_out_ch,
                        **kwargs
                    )

                # After fusion: single fused branch
                else:
                    self._conv_block(
                        seq=self.fused_conv,
                        in_ch=in_ch,
                        out_ch= out_ch,
                        **kwargs
                    )

                in_ch = out_ch

            # Compute flattened dimension
            final_order = order - len(convs)
            n_vertices = Ico.n_vertices(order=final_order)
            self.flat_dim = convs[-1] * n_vertices

            self.fc = nn.Linear(self.flat_dim, latent_dim)

    @classmethod
    def _conv_block(
            cls,
            seq: nn.Sequential,
            level: int,
            standard_ico: bool,
            depth: int,
            in_ch: int,
            out_ch: int,
            order: int,
            batch_norm: bool,
            activation: str,
            n_jobs: int,
        ) -> None:
        """
        Add a single SCNN block to a given sequential container.

        Each block consists of:
        - a DiNe convolution (`IcoConv`)
        - optional batch normalization
        - optional activation function
        - a spherical pooling operator (`IcoPool`)

        Parameters
        ----------
        seq : nn.Sequential
            The module container to which the block is appended.
        level : int
            The block level.
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
        batch_norm : bool
            If True, applies batch normalization after the convolution.
        activation : str
            Name of the activation class from `torch.nn` to apply after each
            convolution layer.
        n_jobs : int
            Number of parallel workers.
            Default 1.
        """
        seq.add_module(
            f"conv_{level}",
            IcoConv(
                in_feats=in_ch,
                out_feats=out_ch,
                order=order,
                depth=depth,
                standard_ico=standard_ico,
                n_jobs=n_jobs,
            ),
        )

        if batch_norm:
            seq.add_module(
                f"bn_{level}",
                nn.BatchNorm1d(out_ch),
            )

        activation_cls = getattr(nn, activation)
        seq.add_module(
            f"act_{level}",
            activation_cls(inplace=True),
        )

        seq.add_module(
            f"pool_{level}",
            IcoPool(
                order=order,
                standard_ico=standard_ico,
                pooling_type="mean",
                n_jobs=n_jobs,
            ),
        )

    def forward(
            self,
            x: tuple[torch.Tensor, torch.Tensor],
        ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : tuple[torch.Tensor, torch.Tensor]
            A tuple ``(left_x, right_x)`` containing the left and right
            cortical textures. Each tensor has shape:
            ``(batch_size, input_channels, n_vertices)``.

        Returns
        -------
        torch.Tensor
            Latent representations of shape ``(batch_size, latent_dim)``.
        """
        left_x, right_x = x

        # Hemisphere-specific feature extraction
        left_x = self.left_conv(left_x)
        right_x = self.right_conv(right_x)

        # Concatenate along channel dimension
        x = torch.cat((left_x, right_x), dim=1)

        # Fused SCNN blocks
        x = self.fused_conv(x)

        # Flatten and project to latent space
        x = x.view(-1, self.flat_dim)
        x = self.fc(x)

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
            Name of of model to be loaded: 'openbhb_simclr_v1'.
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
            If the pretrained model in not recognize.
        """
        if weights_file is None:
            raise NotImplementedError
        if name == "openbhb_simclr_v1":
            model = cls(
                order=5,
                standard_ico=False,
                depth=1,
                input_channels=3,
                latent_dim=128,
                convs=[128, 128, 256, 256],
                fusion_level=1,
                activation="ReLU",
                batch_norm=False,
                cache_file=cache_file,
                printer=printer,
                n_jobs=n_jobs,
            )
        else:
            raise ValueError(
                f"Uknown pretrained model: {name}"
            )
        state = torch.load(weights_file, map_location="cpu")
        model.load_state_dict(state)
        return model
