##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Surface vision transformers.
"""

import fnmatch
from pathlib import Path
from typing import Callable, Self

import numpy as np
import timm
import torch
from torch import nn

from .. import Ico


class SiT(nn.Module):
    """
    Surface Vision Transformer (SiT) [1]_.

    This module implements a Vision Transformer adapted to cortical surface
    data. Each surface patch (n_channels x n_vertices) is flattened and
    projected into a token of dimension ``dim``. A class token is prepended,
    positional embeddings are added, and the resulting sequence is processed
    by a stack of Transformer blocks. The final representation can be pooled
    and optionally passed through a prediction head.

    Parameters
    ----------
    order : int
        Subdivision order of the icosahedron.
        Default 7.
    standard_ico : bool
        If True, recursively subdivides a standard icosahedron.
        If False, loads precomputed FreeSurfer icosahedra.
        Default False.
    dim : int
        Embedding dimension of each token in the Transformer sequence.
        Each flattened patch is projected to this dimension.
        Default 192.
    depth : int
        Number of Transformer blocks. Each block contains a multi-head
        self-attention (MSA) layer and a feed-forward network (FFN).
        Default 12.
    heads : int
        Number of attention heads in each MSA layer.
        Default 3.
    mlp_dim : int
        Hidden dimension of the FFN inside each Transformer block.
        Default 768.
    patch_size : int
        Patch subdivision depth.
        Default 2.
    n_channels : int
        Number of input channels per patch (e.g., thickness, curvature,
        sulcal depth).
        Default 1.
    n_classes: int
        Number of output classes. If ``n_classes <= 0``, the model returns
        the latent embedding instead of a prediction.
        Default 1.
    pool: str
        Pooling strategy applied after the Transformer:
        - ``"cls"``: use the class token representation,
        - ``"mean"``: average all tokens.
        Default ``"cls"``.
    dim_head: int
        Dimension of each attention head.
        Default 64.
    dropout: float
        Dropout rate applied inside Transformer blocks.
        Default 0.
    emb_dropout: float
        Dropout rate applied to the input token embeddings.
        Default 0.
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
    >>> from nidl.surface.backbones import SiT
    >>> model = SiT(
    ...     order=5,
    ... )
    >>> print(model)

    References
    ----------
    .. [1] Dahan, Simon et al., Surface Vision Transformers: Attention-Based
    Modelling applied to Cortical Analysis, MIDL, 2022.
    """
    def __init__(
            self,
            order: int = 5,
            standard_ico: bool = False,
            dim: int = 192,
            depth: int = 12,
            heads: int = 3,
            mlp_dim: int = 768,
            patch_size: int = 2,
            n_channels: int = 1,
            n_classes: int = 1,
            pool: str = "cls",
            dim_head: int = 64,
            dropout: float = 0.,
            emb_dropout: float = 0.,
            cache_file: str | Path | None = None,
            printer: Callable | None = None,
            n_jobs: int = 1,
        ) -> None:
        super().__init__()

        self.order = order
        self.standard_ico = standard_ico
        self.depth = depth
        self.latent_dim = mlp_dim
        self.printer = printer

        assert pool in {"cls", "mean"}, (
            "Pool type must be either cls (cls token) or mean (mean pooling).")

        # Build blocks
        self.printer(
            "SiT initialization may take some time if no caching "
            "initialization is used or the cache is empty."
        )
        with Ico.cachemanager(cache_file):
            self.patches = Ico.patches(
                size=patch_size,
                order=order,
                standard_ico=standard_ico,
                direct_neighbor=False,
                n_jobs=n_jobs,
            )
            self.n_patches, self.n_vertices = self.patches.shape

        self.patch_embedding = nn.Linear(n_channels * self.n_vertices, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_patches + 1, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout,
        )
        self.pool = pool
        self.mlp_head = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, n_classes)) if n_classes > 0
            else nn.Identity()
        )

    def forward(
            self,
            x: torch.Tensor,
        ) -> torch.Tensor:
        """
        Forward method.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, n_channels, n_vertices)``.
            It may contain either left- or right-hemisphere data, but must
            first be symmetrized (as in an xhemi-style dataset).

        Returns
        -------
        torch.Tensor
            If ``n_classes > 0``:
                Output predictions of shape ``(batch_size, n_classes)``.
            If ``n_classes <= 0``:
                Latent embedding of shape ``(batch_size, dim)``.
        """
        # Rearange
        x = x[..., self.patches]
        x = torch.swapdims(x, 1, 2)
        x = torch.flatten(x, start_dim=2)

        # Linear embeding
        x = self.patch_embedding(x)
        n_samples, n_patches, _ = x.shape

        # Positional embeding
        cls_tokens = self.cls_token.repeat(n_samples, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n_patches + 1)]
        x = self.dropout(x)

        # L transformer blocks
        x = self.transformer(x)

        # Pooling
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        # Prediction
        x = self.mlp_head(x)

        return x

    @classmethod
    def from_pretrained(
            cls,
            name: str,
            n_channels: int = 4,
            n_classes: int = 1,
            weights_file: str | Path | None = None,
            cache_file: str | Path | None = None,
            printer: Callable | None = None,
            n_jobs: int = 1,
        ) -> Self:
        """
        Create a model instance and load pretrained weights.

        We fix the patch parameters from the original paper, generating 320
        patches per hemisphere and per channel. Since there are C input
        channels and each patch contains V = 153 vertices, the total
        dimensionality per patch is V x C.

        Parameters
        ----------
        name : str
            Name of of model to be loaded: 'vit_tiny', 'vit_small', or
            'vit_base'.
        n_channels : int
            Number of input channels per patch (e.g., thickness, curvature,
            sulcal depth).
            Default 4.
        n_classes: int
            Number of output classes. If ``n_classes <= 0``, the model returns
            the latent embedding instead of a prediction.
            Default 1.
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
        if name == "vit_tiny":
            model = cls(
                order=6,
                standard_ico=False,
                dim=192,
                depth=12,
                heads=3,
                mlp_dim=768,
                patch_size=4,
                n_channels=n_channels,
                n_classes=n_classes,
                pool="cls",
                dim_head=64,
                dropout=0.,
                emb_dropout=0.,
                cache_file=cache_file,
                printer=printer,
                n_jobs=n_jobs,
            )
        elif name == "vit_small":
            model = cls(
                order=6,
                standard_ico=False,
                dim=384,
                depth=12,
                heads=6,
                mlp_dim=1536,
                patch_size=4,
                n_channels=n_channels,
                n_classes=n_classes,
                pool="cls",
                dim_head=64,
                dropout=0.,
                emb_dropout=0.,
                cache_file=cache_file,
                printer=printer,
                n_jobs=n_jobs,
            )
        elif name == "vit_base":
            model = cls(
                order=6,
                standard_ico=False,
                dim=768,
                depth=12,
                heads=12,
                mlp_dim=3072,
                patch_size=4,
                n_channels=n_channels,
                n_classes=n_classes,
                pool="cls",
                dim_head=64,
                dropout=0.,
                emb_dropout=0.,
                cache_file=cache_file,
                printer=printer,
                n_jobs=n_jobs,
            )
        else:
            raise ValueError(
                f"Uknown pretrainevit_tiny_patch16_224d model: {name}"
            )
        if weights_file is None:
            short_name = f"{name}_patch16_224"
            model_trained = timm.create_model(short_name, pretrained=True)
            state = cls._load_weights_imagenet(
                model.state_dict(), model_trained.state_dict(), 12, 0,
            )
        else:
            state = torch.load(
                weights_file, map_location="cpu", weights_only=False
            )
        model.load_state_dict(state)
        return model

    @classmethod
    def _load_weights_imagenet(
            cls,
            state: dict,
            imagenet_state: dict,
            n_layers: int,
            n_classes: int,
        ) -> dict:
        """
        Load ImageNet-pretrained weights into the model's state dictionary.

        Parameters
        ----------
        state : dict
            Target state dictionary to be populated with transformed weights.
        imagenet_state : dict
            Source state dictionary containing ImageNet-pretrained weights.
        n_layers : int
            Number of transformer layers to load.
        n_classes : int
            Number of output classes. If greater than zero, the prediction head
            weights are also loaded.

        Returns
        -------
        dict
            Updated state dictionary containing the mapped ImageNet weights.

        Notes
        -----
        This method maps weights from a ViT-style ImageNet checkpoint to the
        internal transformer layout used by this model. It handles:
        - LayerNorm mappings
        - Attention projections (qkv, proj)
        - MLP blocks (fc1, fc2)
        - Optional classification head
        """
        # Transformer blocks
        for idx in range(n_layers):
            name = f"transformer.layers.{idx}"
            imagenet_name = f"blocks.{idx}"

            # Norms
            state[f"{name}.0.norm.weight"] = imagenet_state[
                f"{imagenet_name}.norm1.weight"
            ].data
            state[f"{name}.0.norm.bias"] = imagenet_state[
                f"{imagenet_name}.norm1.bias"
            ].data

            # MLP norms
            state[f"{name}.1.net.0.weight"] = imagenet_state[
                f"{imagenet_name}.norm2.weight"
            ].data
            state[f"{name}.1.net.0.bias"] = imagenet_state[
                f"{imagenet_name}.norm2.bias"
            ].data

            # Attention
            state[f"{name}.0.to_qkv.weight"] = imagenet_state[
                f"{imagenet_name}.attn.qkv.weight"
            ].data
            state[f"{name}.0.to_out.0.weight"] = imagenet_state[
                f"{imagenet_name}.attn.proj.weight"
            ].data
            state[f"{name}.0.to_out.0.bias"] = imagenet_state[
                f"{imagenet_name}.attn.proj.bias"
            ].data

            # MLP
            state[f"{name}.1.net.1.weight"] = imagenet_state[
                f"{imagenet_name}.mlp.fc1.weight"
            ].data
            state[f"{name}.1.net.1.bias"] = imagenet_state[
                f"{imagenet_name}.mlp.fc1.bias"
            ].data
            state[f"{name}.1.net.4.weight"] = imagenet_state[
                f"{imagenet_name}.mlp.fc2.weight"
            ].data
            state[f"{name}.1.net.4.bias"] = imagenet_state[
                f"{imagenet_name}.mlp.fc2.bias"
            ].data

        # Prediction head
        if n_classes > 0:
            state["mlp_head.0.weight"] = imagenet_state["norm.weight"].data
            state["mlp_head.0.bias"] = imagenet_state["norm.bias"].data

        return state


class SiTAttentionMaps(nn.Module):
    """
    Minimal standalone extractor for attention maps using forward hooks.

    Parameters
    ----------
    model : torch.nn.Module
        Model from which attention maps will be extracted.
    names : tuple[str]
        List of module names or wildcard patterns to hook.
        Default ('*attend').

    Returns
    -------
    numpy.ndarray
        Concatenated attention maps of shape ``(N, ...)`` where ``N`` is the
        total number of hooked modules.

    Notes
    -----
    - Uses forward hooks only.
    - All matched modules are hooked.
    - The model must return tensors compatible with NumPy conversion.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            names: tuple[str] = ("*attend", ),
        ) -> None:
        super().__init__()

        self.model = model
        self.names = names
        self.hooks = []
        self.outputs = []

        # Resolve module names
        module_names = [name for name, _ in model.named_modules()]
        matched = []
        for pattern in names:
            matched.extend(fnmatch.filter(module_names, pattern))
        if not matched:
            raise RuntimeError(f"No modules matched patterns: {names}")

        # Register hooks
        for name, module in model.named_modules():
            if name in matched:
                self.hooks.append(
                    module.register_forward_hook(self._hook)
                )

    def _hook(self, module, inp, out):
        """
        Store hook output as NumPy.
        """
        self.outputs.append(out.detach().cpu().numpy())

    def forward(
            self,
            x: torch.Tensor,
            mask: np.ndarray | None = None,
        ) -> np.ndarray:
        """
        Run the model, collect attention maps, and project them onto the
        spherical vertex domain.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, n_channels, n_vertices)``.
        mask : numpy.ndarray
            Binary mask of shape ``(n_vertices,)`` used to zero out invalid
            vertices.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_vertices, n_layers, n_heads)`` containing
            per-vertex attention values.
        """
        # Run model and collect hook outputs
        self.outputs.clear()
        _ = self.model(x)
        out_maps = np.concatenate(self.outputs, axis=0)

        # Extract layer/head structure
        n_layers, n_heads = out_maps.shape[:2]

        # Keep only attention from CLS to patches
        # shape becomes (n_layers, n_heads, n_patches)
        attentions = out_maps[:, :, 1:, 0]

        # Reshape to (n_patches, n_layers, n_heads)
        attentions = attentions.transpose(2, 0, 1)

        # Allocate spherical texture
        n_vertices = Ico.n_vertices(order=self.model.order)
        attention_texture = np.zeros(
            (n_vertices, n_layers, n_heads),
            dtype=np.float32,
        )

        # Scatter patch attention into vertex domain
        for idx in range(self.model.n_patches):
            indices = self.model.patches[idx]
            attention_texture[indices] = attentions[idx]

        # Apply mask
        if mask is not None:
            attention_texture *= mask[:, None, None]

        return attention_texture


class FeedForward(nn.Module):
    """
    Transformer feed-forward network.

    This module implements the MLP block used inside Transformer layers.
    It applies:
    - LayerNorm,
    - a linear projection from ``dim`` to ``hidden_dim``,
    - a GELU activation,
    - dropout,
    - a projection back to ``dim``.

    Parameters
    ----------
    dim : int
        Input and output feature dimension.
    hidden_dim : int
        Hidden expansion dimension inside the MLP.
    dropout : float
        Dropout rate applied after each linear layer.
        Default is 0.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(..., dim)`` after feed-forward transformation.
    """
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            dropout: float = 0.,
        ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(
            self,
            x: torch.Tensor,
        ) -> torch.Tensor:
        """
        Apply the feed-forward transformation.
        """
        return self.net(x)


class Attention(nn.Module):
    """
    Multi-head self-attention mechanism.

    This module computes scaled dot-product attention across a sequence.
    It performs:
    - LayerNorm,
    - linear projection to queries/keys/values,
    - multi-head attention,
    - optional output projection.

    Parameters
    ----------
    dim : int
        Input embedding dimension.
    heads : int
        Number of attention heads.
        Default is 8.
    dim_head : int
        Dimension of each attention head.
        Default is 64.
    dropout : float
        Dropout rate applied to attention weights and output projection.
        Default is 0.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(batch, seq_len, dim)`` after attention.
    """
    def __init__(
            self,
            dim: int,
            heads: int = 8,
            dim_head: int = 64,
            dropout: float = 0.,
        ) -> None:
        super().__init__()

        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
        ) -> torch.Tensor:
        """
        Compute multi-head self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, seq_len, dim)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch, seq_len, dim)``.
        """
        x = self.norm(x)

        # Compute Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (torch.swapdims(
            torch.reshape(t, (t.size(dim=0), t.size(dim=1), self.heads, -1)),
            1, 2) for t in qkv)

        # Scaled dot-product attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # Weighted sum of values
        out = torch.matmul(attn, v)
        out = torch.swapdims(out, 1, 2)
        out = torch.flatten(out, start_dim=2)

        return self.to_out(out)


class Transformer(nn.Module):
    """
    Transformer encoder composed of multiple layers.

    Each layer contains:
    - a multi-head self-attention block,
    - a feed-forward MLP block,
    both wrapped in residual connections.

    Parameters
    ----------
    dim : int
        Embedding dimension of each token.
    depth : int
        Number of Transformer layers.
    heads : int
        Number of attention heads per layer.
    dim_head : int
        Dimension of each attention head.
    mlp_dim : int
        Hidden dimension of the feed-forward network.
    dropout : float, optional
        Dropout rate applied inside attention and MLP blocks.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(batch, seq_len, dim)`` after the full encoder.
    """
    def __init__(
            self,
            dim: int,
            depth: int,
            heads: int,
            dim_head: int,
            mlp_dim: int,
            dropout: float = 0.,
        ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(
                    dim, heads=heads, dim_head=dim_head, dropout=dropout
                ),
                FeedForward(dim, mlp_dim, dropout=dropout),
            ])
            for _ in range(depth)
        ])

    def forward(
            self,
            x: torch.Tensor,
        ) -> torch.Tensor:
        """
        Apply a stack of Transformer layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, seq_len, dim)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch, seq_len, dim)``.
        """
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
