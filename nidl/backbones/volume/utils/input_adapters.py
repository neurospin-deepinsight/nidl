##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn.functional as func
from einops import rearrange
from torch import nn

from .pos_embed import build_3d_sincos_posemb


def triple(t):
    return t if isinstance(t, tuple) else (t, t, t)


class SimpleFeaturesInputAdapter(nn.Module):
    """Adapter for 1D vectors, like tabular data without learnable weights.

    It directly maps the input vector to a 1-length sequence.

    Adapted from https://github.com/Duplums/CoMM.
    """
    def forward(self, x):
        assert x.ndim == 2, f"Expected 2D tensors, got {x.shape}"
        x = rearrange(x, 'b n -> b 1 n')
        return x


class IdentityAdapter(nn.Module):
    """Adapter for transformer encoders, whose output is already a sequence
    of tokens (the expert scores of MoE encoders are discarded).

    Adapted from https://github.com/Duplums/CoMM.
    """

    def forward(self, x):
        # For Moe encoders, discard experts scores
        if isinstance(x, tuple):
            x = x[0]

        assert x.ndim == 3, (
            f"Expected 3D tensors of shape (B, N, E), got {x.shape}")
        return x


class Patched3DInputAdapter(nn.Module):
    """Adapter for 3D spatial inputs, like volumetric images or feature maps.

    Creates tokens from patches over the volume.

    Volumes are laid out as (B, C, Z, H, W), matching Conv3d and the
    (b, d, z, h, w) layout of `build_3d_sincos_posemb`.

    Adapted from the 2D PatchedInputAdapter of
    https://github.com/Duplums/CoMM.

    Parameters
    ----------
    num_channels : int
        Number of input channels of the volume.
    stride_level : int
        Stride level compared to the full-sized volume. E.g. 4 for 1/4th the
        size of the volume.
    patch_size_full : int or tuple of (int, int, int)
        Patch size over the full volume size. Patch size for smaller inputs
        will be computed accordingly.
    dim_tokens : int or None, default=None
        Dimension of output tokens. Can be set using the :meth:`init` method.
    sincos_pos_emb : bool, default=True
        Set to True to use fixed 3D sin-cos positional embeddings.
    learnable_pos_emb : bool, default=False
        Set to True to learn positional embeddings instead.
    volume_size : int or tuple of (int, int, int), default=128
        Default volume size. Used to initialize size of positional
        embeddings.
    """
    def __init__(self,
                 num_channels: int,
                 stride_level: int,
                 patch_size_full: Union[int, tuple[int, int, int]],
                 dim_tokens: Optional[int] = None,
                 sincos_pos_emb: bool = True,
                 learnable_pos_emb: bool = False,
                 volume_size: Union[int, tuple[int, int, int]] = 128):

        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size_full = triple(patch_size_full)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.learnable_pos_emb = learnable_pos_emb
        self.volume_size = triple(volume_size)
        self.num_patches = ((self.volume_size[0] // self.patch_size_full[0])
                            * (self.volume_size[1] // self.patch_size_full[1])
                            * (self.volume_size[2] // self.patch_size_full[2]))

        # Actual patch size per axis, taking the stride of the input into
        # account
        self.P_Z = max(1, self.patch_size_full[0] // stride_level)
        self.P_H = max(1, self.patch_size_full[1] // stride_level)
        self.P_W = max(1, self.patch_size_full[2] // stride_level)

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768):
        """Initialize parts of the adapter that depend on the dimension of
        tokens.

        Parameters
        ----------
        dim_tokens : int, default=768
            Dimension of tokens.
        """
        self.dim_tokens = dim_tokens

        # Fixed-size positional embeddings, interpolated to the actual grid
        # in forward
        z_posemb = self.volume_size[0] // (self.stride_level * self.P_Z)
        h_posemb = self.volume_size[1] // (self.stride_level * self.P_H)
        w_posemb = self.volume_size[2] // (self.stride_level * self.P_W)
        if self.sincos_pos_emb:
            self.pos_emb = build_3d_sincos_posemb(
                z_posemb, h_posemb, w_posemb, embed_dim=self.dim_tokens)
            self.pos_emb = nn.Parameter(
                self.pos_emb, requires_grad=self.learnable_pos_emb)
        else:
            self.pos_emb = nn.Parameter(
                torch.zeros(1, self.dim_tokens, z_posemb, h_posemb, w_posemb))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Volume -> tokens projection
        self.proj = nn.Conv3d(
            in_channels=self.num_channels, out_channels=self.dim_tokens,
            kernel_size=(self.P_Z, self.P_H, self.P_W),
            stride=(self.P_Z, self.P_H, self.P_W)
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_emb'}

    def forward(self, x):
        """Forward pass through the input adapter, transforming a volume into
        a sequence of tokens and adding positional encodings.

        Parameters
        ----------
        x : torch.Tensor
            Input volume tensor of shape (B, C, Z, H, W).

        Returns
        -------
        x : torch.Tensor
            Tokens of shape (B, N_Z * N_H * N_W, dim_tokens).
        """
        _, _, Z, H, W = x.shape
        assert self.dim_tokens is not None, (
            'Need to call init(dim_tokens) function first')
        assert (
            (Z % self.P_Z == 0) and (H % self.P_H == 0) and (W % self.P_W == 0)
        ), (
            f'Volume sizes {Z}x{H}x{W} must be divisible by patch sizes '
            f'{self.P_Z}x{self.P_H}x{self.P_W}')
        # Number of patches along each axis
        N_Z, N_H, N_W = Z // self.P_Z, H // self.P_H, W // self.P_W

        # Create patches [B, C, Z, H, W] -> [B, (N_Z*N_H*N_W), D]
        x_patch = rearrange(self.proj(x), 'b d nz nh nw -> b (nz nh nw) d')

        # Create positional embedding
        x_pos_emb = func.interpolate(self.pos_emb, size=(N_Z, N_H, N_W),
                                     mode='trilinear', align_corners=False)
        x_pos_emb = rearrange(x_pos_emb, 'b d nz nh nw -> b (nz nh nw) d')

        # Add patches and positional embeddings
        x = x_patch + x_pos_emb

        return x
