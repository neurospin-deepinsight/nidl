##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

import math

import numpy as np
import torch
from einops import rearrange


def build_1d_sincos_posemb(max_seq_len: int, embed_dim: int = 1024):
    """Build positional embedding matrix with `batch_first`.

    Parameters
    ----------
    max_seq_len: int
        Maximum length of the input sequence.

    embed_dim: int, default=1024
        Dimension of each token.

    Returns
    ----------
    pe: torch.Tensor with shape (1, max_seq_length, embed_dim)
    """
    if embed_dim % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            f"odd dim (got dim={embed_dim:d})"
        )
    pe = torch.zeros(max_seq_len, embed_dim)
    position = torch.arange(0, max_seq_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float)
        * -(math.log(10000.0) / embed_dim)
    )
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    pe = pe[None, :, :]
    return pe


def build_2d_sincos_posemb(h, w, embed_dim=1024, temperature=10000.0):
    """Sine-cosine positional embeddings from MoCo-v3

    Source: https://github.com/facebookresearch/moco-v3/blob/main/vits.py
    """
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
    assert embed_dim % 4 == 0, (
        "Embed dimension must be divisible by 4 for 2D sin-cos position "
        "embedding"
    )
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature**omega)
    out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
    out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
    pos_emb = torch.cat(
        [
            torch.sin(out_w),
            torch.cos(out_w),
            torch.sin(out_h),
            torch.cos(out_h),
        ],
        dim=1,
    )[None, :, :]
    pos_emb = rearrange(pos_emb, "b (h w) d -> b d h w", h=h, w=w, d=embed_dim)
    return pos_emb


def build_3d_sincos_posemb(h, w, d, embed_dim=1024, temperature=10000.0):
    """3d sin-cosine positional embeddings from MoCo-v3.

    Source: https://github.com/facebookresearch/moco-v3/blob/main/vits.py and
    https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py
    """
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_d = torch.arange(d, dtype=torch.float32)
    grid_w, grid_h, grid_d = torch.meshgrid(
        grid_w, grid_h, grid_d, indexing="ij"
    )
    # Trick from https://github.com/tatp22/multidim-positional-encoding
    # We take a larger 'embed_dim' if not divisible by 6
    pos_dim = int(np.ceil(embed_dim / 6))
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature**omega)
    out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
    out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
    out_d = torch.einsum("m,d->md", [grid_d.flatten(), omega])
    pos_emb = torch.cat(
        [
            torch.sin(out_w),
            torch.cos(out_w),
            torch.sin(out_h),
            torch.cos(out_h),
            torch.sin(out_d),
            torch.cos(out_d),
        ],
        dim=1,
    )[None, :, :]  # add batch dim
    pos_emb = rearrange(
        pos_emb[:, :, :embed_dim],
        "b (h w d) l -> b l h w d",  # truncate along the embedding dimension
        h=h,
        w=w,
        d=d,
        l=embed_dim,
    )
    return pos_emb
