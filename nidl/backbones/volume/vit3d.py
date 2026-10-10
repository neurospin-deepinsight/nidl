##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Callable, Optional, Union

import torch
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Attention, Block, Mlp
from torch import Tensor, nn

from .utils.pos_embed import build_3d_sincos_posemb


def _to_3tuple(x: int | Sequence[int]) -> tuple[int, int, int]:
    """
    Convert an integer or a length-3 sequence to a 3-tuple.

    Parameters
    ----------
    x : int or sequence of int
        Input value.

    Returns
    -------
    tuple of int
        Tuple in ``(H, W, D)`` order.
    """
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        if len(x) != 3:
            raise ValueError(f"Expected length-3 sequence, got {x}.")
        return int(x[0]), int(x[1]), int(x[2])
    v = int(x)
    return v, v, v


def _assert_divisible(
    img_size: tuple[int, int, int], patch_size: tuple[int, int, int]
) -> None:
    """
    Validate that the image size is divisible by patch size.

    Parameters
    ----------
    img_size : tuple of int
        Input size in ``(H, W, D)``.
    patch_size : tuple of int
        Patch size in ``(H, W, D)``.

    Raises
    ------
    ValueError
        If one dimension is not divisible.
    """
    if any(i % p != 0 for i, p in zip(img_size, patch_size)):
        raise ValueError(
            f"img_size={img_size} must be divisible by "
            f"patch_size={patch_size}."
        )


class PatchEmbed3D(nn.Module):
    """
    3D patch embedding with a strided 3D convolution.

    Parameters
    ----------
    img_size : int or sequence of int
        Input volume size in ``(H, W, D)``.
    patch_size : int or sequence of int
        Patch size in ``(H, W, D)``. Image size must
        be divisible by patch size in each direction.
    in_chans : int, default=1
        Number of input channels.
    embed_dim : int, default=768
        Embedding dimension.
    bias : bool, default=True
        Whether to use bias in the projection layer.
    strict_img_size : bool, default=True
        Whether to require exact input size during forward.
    flatten : bool, default=True
        Whether to flatten to token sequence.
    output_fmt : str, default="NLC"
        Output format if ``flatten=True``. Only ``"NLC"`` is supported.

    Attributes
    ----------
    img_size : tuple of int
        Input size in ``(H, W, D)``.
    patch_size : tuple of int
        Patch size in ``(H, W, D)``.
    grid_size : tuple of int
        Patch grid size in ``(H, W, D)``.
    num_patches : int
        Number of patches.
    """

    def __init__(
        self,
        img_size: int | Sequence[int],
        patch_size: int | Sequence[int],
        in_chans: int = 1,
        embed_dim: int = 768,
        bias: bool = True,
        strict_img_size: bool = True,
        flatten: bool = True,
        output_fmt: str = "NLC",
    ) -> None:
        super().__init__()
        self.img_size = _to_3tuple(img_size)
        self.patch_size = _to_3tuple(patch_size)
        _assert_divisible(self.img_size, self.patch_size)

        self.grid_size = tuple(
            i // p for i, p in zip(self.img_size, self.patch_size)
        )
        self.num_patches = (
            self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        )
        self.strict_img_size = strict_img_size
        self.flatten = flatten
        self.output_fmt = output_fmt

        if self.flatten and self.output_fmt != "NLC":
            raise ValueError(
                "Only output_fmt='NLC' is supported when flatten=True."
            )

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Embed a 3D volume into patch tokens.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, C, H, W, D)``.

        Returns
        -------
        torch.Tensor
            If ``flatten=True``, patch tokens of shape ``(B, N, C)``.
            Otherwise, projected tensor of shape ``(B, C, H', W', D')``.

        Raises
        ------
        ValueError
            If input size is invalid when ``strict_img_size=True``.
        """
        if self.strict_img_size and tuple(x.shape[-3:]) != self.img_size:
            raise ValueError(
                f"Expected input size {self.img_size}, got "
                f"{tuple(x.shape[-3:])}."
            )
        if len(x.shape) != 5:
            raise ValueError(
                f"Expected input shape (B, C, H, W, D), got {x.shape}."
            )

        x = self.proj(x)
        if not self.flatten:
            return x

        # (B, C, H', W', D') -> (B, N, C)
        return x.flatten(2).transpose(1, 2)


class VisionTransformer3D(nn.Module):
    """
    3D Vision Transformer with a timm-like interface.

    Parameters
    ----------
    img_size : int or sequence of int
        Input size in ``(H, W, D)``.
    patch_size : int or sequence of int
        Patch size in ``(H, W, D)``.
    in_chans : int, default=1
        Number of input channels.
    num_classes : int, default=0
        Number of output classes. If non-positive, head is identity.
    global_pool : {"cls_token", "avg", "max", "avgmax", ""},
        default="cls_token"
        Pooling mode. If "", no pooling is applied and the full token sequence
        is returned.
    embed_dim : int, default=768
        Embedding dimension.
    depth : int, default=12
        Number of transformer blocks.
    num_heads : int, default=12
        Number of attention heads.
    mlp_ratio : float, default=4.0
        MLP expansion ratio.
    qkv_bias : bool, default=True
        Whether to use qkv bias.
    class_token : bool, default=True
        Whether to prepend a CLS token.
    reg_tokens : int, default=0
        Number of register tokens.
    no_embed_class : bool, default=False
        If ``True``, position embeddings are defined only on patch tokens.
    pre_norm : bool, default=False
        Whether to apply normalization before the transformer blocks.
    fc_norm : bool or None, default=None
        Whether to apply normalization after pooling. If ``None``, defaults
        to ``global_pool == "avg"``.
    dynamic_img_size : bool, default=False
        Present for interface compatibility. Not used here.
    pos_embed : {"learned", "sincos", "none"}, default="learned"
        Absolute positional embedding mode.
    drop_rate : float, default=0.0
        Head dropout.
    pos_drop_rate : float, default=0.0
        Positional dropout.
    proj_drop_rate : float, default=0.0
        Projection dropout in each transformer block.
    attn_drop_rate : float, default=0.0
        Attention dropout in each transformer block.
    drop_path_rate : float, default=0.0
        Maximum stochastic depth rate.
    norm_layer : callable, default=nn.LayerNorm
        Normalization layer constructor.
    act_layer : callable, default=nn.GELU
        Activation layer constructor.

    Notes
    -----
    This implementation reuses timm's transformer `Block` as-is.
    Only the 3D-specific pieces are implemented locally:
    patch embedding, 3D positional embeddings, and checkpoint inflation.
    """

    def __init__(
        self,
        img_size: Union[int, Sequence[int]],
        patch_size: Union[int, Sequence[int]],
        in_chans: int = 1,
        num_classes: int = 0,
        global_pool: str = "cls_token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        class_token: bool = True,
        reg_tokens: int = 0,
        no_embed_class: bool = False,
        pre_norm: bool = False,
        final_norm: bool = True,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        pos_embed: str = "learned",
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        embed_layer: Callable = PatchEmbed3D,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        block_fn: Callable[..., nn.Module] = Block,
        mlp_layer: Callable[..., nn.Module] = Mlp,
        attn_layer=Attention,
    ) -> None:
        super().__init__()

        self.num_classes = int(num_classes)
        self.global_pool = global_pool
        self.embed_dim = embed_dim
        self.has_class_token = class_token
        self.num_reg_tokens = int(reg_tokens)
        self.no_embed_class = bool(no_embed_class)
        self.dynamic_img_size = dynamic_img_size
        self.num_prefix_tokens = (
            1 if class_token else 0
        ) + self.num_reg_tokens
        self.num_features = embed_dim
        self.head_hidden_size = embed_dim

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,
            strict_img_size=not dynamic_img_size,
        )
        self.grid_size = self.patch_embed.grid_size
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        )
        self.reg_token = (
            nn.Parameter(torch.zeros(1, self.num_reg_tokens, embed_dim))
            if self.num_reg_tokens > 0
            else None
        )

        if pos_embed == "learned":
            pos_len = (
                self.num_patches
                if self.no_embed_class
                else self.num_patches + self.num_prefix_tokens
            )
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_len, embed_dim))
        elif pos_embed == "sincos":
            pos = build_3d_sincos_posemb(*self.grid_size, embed_dim=embed_dim)
            pos = pos.flatten(2).transpose(1, 2)  # (1, N_patches, C)
            if not self.no_embed_class and self.num_prefix_tokens > 0:
                prefix = torch.zeros(
                    1, self.num_prefix_tokens, embed_dim, dtype=pos.dtype
                )
                pos = torch.cat([prefix, pos], dim=1)
            self.register_buffer("pos_embed", pos, persistent=False)
        elif pos_embed == "none":
            self.pos_embed = None
        else:
            raise ValueError(f"Unsupported pos_embed mode: {pos_embed}.")

        self.pos_drop = nn.Dropout(pos_drop_rate)
        self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    scale_attn_norm=scale_attn_norm,
                    scale_mlp_norm=scale_mlp_norm,
                    proj_bias=proj_bias,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                    attn_layer=attn_layer,
                    depth=i,
                )
                for i in range(depth)
            ]
        )
        use_fc_norm = (
            global_pool in ("avg", "avgmax", "max")
            if fc_norm is None
            else fc_norm
        )
        self.norm = (
            norm_layer(embed_dim)
            if final_norm and not use_fc_norm
            else nn.Identity()
        )
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.init_weights()

    def init_weights(self) -> None:
        """
        Initialize model weights.
        """
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=0.02)
        if self.reg_token is not None:
            trunc_normal_(self.reg_token, std=0.02)
        if isinstance(self.pos_embed, nn.Parameter):
            trunc_normal_(self.pos_embed, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _pos_embed(self, x: Tensor) -> Tensor:
        """
        Add prefix tokens and absolute position embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Patch tokens of shape ``(B, N, C)``.

        Returns
        -------
        torch.Tensor
            Tokens after prefix concatenation and positional embedding.
        """
        if len(x.shape) != 3:
            raise ValueError(f"Expected input shape (B, N, C), got {x.shape}.")

        B = x.shape[0]
        prefix = []
        if self.cls_token is not None:
            prefix.append(self.cls_token.expand(B, -1, -1))
        if self.reg_token is not None:
            prefix.append(self.reg_token.expand(B, -1, -1))

        if self.no_embed_class:
            if self.pos_embed is not None:
                if x.shape[1] != self.pos_embed.shape[1]:
                    raise ValueError(
                        f"Expected {self.pos_embed.shape[1]} patch tokens, got"
                        f" {x.shape[1]}."
                    )
                x = x + self.pos_embed
            if prefix:
                x = torch.cat([*prefix, x], dim=1)
        else:
            if prefix:
                x = torch.cat([*prefix, x], dim=1)
            if self.pos_embed is not None:
                if x.shape[1] != self.pos_embed.shape[1]:
                    raise ValueError(
                        f"Expected {self.pos_embed.shape[1]} tokens, got "
                        f"{x.shape[1]}."
                    )

                x = x + self.pos_embed

        return self.pos_drop(x)

    def _patch_tokens(self, x: Tensor) -> Tensor:
        """
        Slice patch tokens from a token sequence.

        Parameters
        ----------
        x : torch.Tensor
            Full token sequence of shape ``(B, N_total, C)``.

        Returns
        -------
        torch.Tensor
            Patch tokens of shape ``(B, N_patches, C)``.
        """
        return x[:, self.num_prefix_tokens :]

    def pool(self, x: Tensor, pool_type: Optional[str] = None) -> Tensor:
        """
        Pool token features.

        Parameters
        ----------
        x : torch.Tensor
            Token features of shape ``(B, N_total, C)``.
        pool_type : str, optional
            Pooling mode. If ``None``, ``self.global_pool`` is used.

        Returns
        -------
        torch.Tensor
            Pooled features of shape ``(B, C)`` or the full sequence if
            ``pool_type == ""``.
        """
        pool_type = self.global_pool if pool_type is None else pool_type

        if pool_type == "":
            return x
        if pool_type == "cls_token":
            if not self.has_class_token:
                raise ValueError(
                    "global_pool='token' requires class_token=True."
                )
            return x[:, 0]

        patch_tokens = self._patch_tokens(x)
        if pool_type == "avg":
            return patch_tokens.mean(dim=1)
        if pool_type == "max":
            return patch_tokens.max(dim=1).values
        if pool_type == "avgmax":
            return 0.5 * (
                patch_tokens.mean(dim=1) + patch_tokens.max(dim=1).values
            )

        raise ValueError(f"Unsupported pool_type: {pool_type}.")

    def forward_features(self, x: Tensor) -> Tensor:
        """
        Compute token features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, C, H, W, D)``.

        Returns
        -------
        torch.Tensor
            Token features of shape ``(B, N_total, C)``.
        """
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        """
        Apply pooling and classification head.

        Parameters
        ----------
        x : torch.Tensor
            Token features of shape ``(B, N_total, C)``.
        pre_logits : bool, default=False
            Whether to return pooled features before the classifier.

        Returns
        -------
        torch.Tensor
            Model outputs.
        """
        x = self.pool(x)
        if x.ndim != 2:
            raise ValueError(
                "forward_head expects pooled features of shape (B, C)."
            )
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, C, H, W, D)``.

        Returns
        -------
        torch.Tensor
            Logits or features depending on the classifier head.
        """
        return self.forward_head(self.forward_features(x))

    def forward_intermediates(
        self,
        x: Tensor,
        indices: Sequence[int] | int = (-1,),
        norm: bool = False,
        output_fmt: str = "NLC",
        intermediates_only: bool = False,
    ) -> tuple[Tensor, list[Tensor]] | list[Tensor]:
        """
        Forward and return intermediate block outputs.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, C, H, W, D)``.
        indices : sequence of int or int, default=(-1,)
            Block indices to return. Negative indices are supported.
        norm : bool, default=False
            Whether to apply final normalization to returned intermediates.
        output_fmt : {"NLC"}, default="NLC"
            Output format for intermediates.
        intermediates_only : bool, default=False
            If ``True``, return only the intermediate tensors.

        Returns
        -------
        tuple or list
            If ``intermediates_only=False``, returns ``(final, intermediates)``
            Otherwise returns ``intermediates``.

        Raises
        ------
        ValueError
            If an unsupported output format is requested.
        """
        if output_fmt != "NLC":
            raise ValueError("Only output_fmt='NLC' is supported.")

        if isinstance(indices, int):
            indices = (indices,)

        n_blocks = len(self.blocks)
        take = {i if i >= 0 else n_blocks + i for i in indices}

        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        intermediates = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take:
                y = self.norm(x) if norm else x
                intermediates.append(y)

        x = self.norm(x)
        if intermediates_only:
            return intermediates
        return x, intermediates

    def reset_classifier(
        self, num_classes: int, global_pool: Optional[str] = None
    ) -> None:
        """
        Reset the classification head.

        Parameters
        ----------
        num_classes : int
            New number of classes.
        global_pool : str, optional
            New global pooling mode.
        """
        self.num_classes = int(num_classes)
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = (
            nn.Linear(self.embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def get_classifier(self) -> nn.Module:
        """
        Return the classifier head.

        Returns
        -------
        nn.Module
            Classifier head.
        """
        return self.head

    def no_weight_decay(self) -> set[str]:
        """
        Return parameter names that should typically avoid weight decay.

        Returns
        -------
        set of str
            Parameter names exempt from weight decay.
        """
        nwd = set()
        if self.cls_token is not None:
            nwd.add("cls_token")
        if self.reg_token is not None:
            nwd.add("reg_token")
        if isinstance(self.pos_embed, nn.Parameter):
            nwd.add("pos_embed")
        return nwd

class DynamicSizeViT(VisionTransformer3D):
    """
    Extension of the 3D ViT backbone that can handle variable-sized inputs
    via positional embeddings interpolation as in DINO [1].
    Positional embeddings are first computed for a grid of shape A.
    When a volume with a grid of shape A' is presented in input pos embeddings
    of shape A' are interpolated from those of shape A.

    The code for positional embeddings interpolation is derived from [2]
    with the adaptations to handle 3D volumes.

    References
        ----------
    [1] Caron, M., et al., "Emerging Properties in Self-Supervised Vision
        Transformers" ICCV, 2021. https://arxiv.org/abs/2104.14294

    [2] https://github.com/facebookresearch/dino/blob/main/vision_transformer.py#L134
    """
    def __init__(
            self,
            img_size,
            patch_size,
            in_chans = 1,
            num_classes = 0,
            global_pool = "cls_token",
            embed_dim = 768,
            depth = 12,
            num_heads = 12,
            mlp_ratio = 4,
            qkv_bias = True,
            qk_norm = False,
            scale_attn_norm = False,
            scale_mlp_norm = False,
            proj_bias = True,
            class_token = True,
            reg_tokens = 0,
            no_embed_class = False,
            pre_norm = False,
            final_norm = True,
            fc_norm = None,
            dynamic_img_size = False,
            pos_embed = "learned",
            drop_rate = 0,
            pos_drop_rate = 0,
            proj_drop_rate = 0,
            attn_drop_rate = 0,
            drop_path_rate = 0,
            embed_layer = PatchEmbed3D,
            norm_layer = nn.LayerNorm,
            act_layer = nn.GELU,
            block_fn = Block,
            mlp_layer = Mlp,
            attn_layer=Attention
        ):
            super().__init__(
                img_size, patch_size, in_chans, num_classes,
                global_pool, embed_dim, depth, num_heads,
                mlp_ratio, qkv_bias, qk_norm, scale_attn_norm,
                scale_mlp_norm, proj_bias, class_token, reg_tokens,
                no_embed_class, pre_norm, final_norm, fc_norm,
                dynamic_img_size, pos_embed, drop_rate, pos_drop_rate,
                proj_drop_rate, attn_drop_rate, drop_path_rate, embed_layer,
                norm_layer, act_layer, block_fn, mlp_layer, attn_layer
            )

    def forward_features(self, x: Tensor) -> Tensor:
        # Shape must be unpacked for resizing
        h, w, d = x.shape[2], x.shape[3], x.shape[4]
        x = self.patch_embed(x)
        x = self._pos_embed(x, h, w, d)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def _pos_embed(self, x: Tensor, h, w, d) -> Tensor:

        if len(x.shape) != 3:
            raise ValueError(f"Expected input shape (B, N, C), got {x.shape}.")

        B = x.shape[0]
        prefix = []
        if self.cls_token is not None:
            prefix.append(self.cls_token.expand(B, -1, -1))
        if self.reg_token is not None:
            prefix.append(self.reg_token.expand(B, -1, -1))

        if (not self.no_embed_class) and prefix:
            # prefix tokens must be included in pos embeddings computation
            x = torch.cat([*prefix, x], dim=1)
        
        # If pos embeddings have been computed for a grid of different shape
        # they are used to interpolate embeddings for new grid shape
        if self.pos_embed is not None:
            if x.shape[1] != self.pos_embed.shape[1]:
                interpolated_embed = self.interpolate_pos_encoding(h, w, d)
            else:
                interpolated_embed = self.pos_embed
                
            x = x + interpolated_embed
    
        if self.no_embed_class and prefix:
            x = torch.cat([*prefix, x], dim=1)

        return self.pos_drop(x)

    # "Resize" of the patch tokens grid to accout
    # for smaller local crops and correct positional embeddings
    def interpolate_pos_encoding(self, h, w, d):
        """
        Interpolates positional embeddings for new target shape from
        positional embeddings computed for a different shape.
        In DINO, positional embeddings computed for (bigger) global crops
        are used to derive embeddings for (smaller) local crops.
        On the 3D volumes, trilinear interpolation is used
        (rather then bicubic as in the original implementation).
        """

        if self.no_embed_class:
            patch_pos_embed = self.pos_embed
        else:
            prefix_pos_embed = self.pos_embed[:, :self.num_prefix_tokens]
            patch_pos_embed = self.pos_embed[:, self.num_prefix_tokens:]

        # original grid size
        w1, h1, d1 = self.grid_size

        # target grid size
        w0 = w // self.patch_embed.patch_size[1]
        h0 = h // self.patch_embed.patch_size[0]
        d0 = d // self.patch_embed.patch_size[2]
        patch_embedding_dim = self.pos_embed.shape[-1]
        # to avoid floating point error in the interpolation
        w0, h0, d0 = w0 + 0.1, h0 + 0.1, d0 + 0.1

        # Reconstruct 2D patch grid in (B, W, H, D, C)
        patches_grid = patch_pos_embed.reshape(
            1,
            h1, w1, d1,
            patch_embedding_dim
        )

        # Interpolate positional embeddings for a grid of shape target size
        # starting from patches_grid embeddings with trilinear interpolation
        patch_pos_embed = nn.functional.interpolate(
            patches_grid.permute(0, 4, 1, 2, 3),  # convert to (B, C, H, W, D)
            scale_factor=(h0 / h1, w0 / w1, d0 / d1),
            mode='trilinear',
            align_corners = False
        )

        # Reshape to 1-D vector of patch embeddings
        patch_pos_embed = patch_pos_embed \
            .permute(0, 2, 3, 4, 1) \
            .view(1, -1, patch_embedding_dim)

        if self.no_embed_class:
            return patch_pos_embed
        else:
            return torch.cat((prefix_pos_embed, patch_pos_embed), dim=1)

def _resize_pos_embed_2d_to_3d(
    pos_embed: Tensor,
    *,
    num_prefix_tokens: int,
    old_grid_size: tuple[int, int],
    new_grid_size: tuple[int, int, int],
) -> Tensor:
    """
    Resize a 2D positional embedding to a 3D grid.

    Parameters
    ----------
    pos_embed : torch.Tensor
        Input positional embedding of shape ``(1, N_old, C)``.
    num_prefix_tokens : int
        Number of prefix tokens.
    old_grid_size : tuple of int
        Original 2D grid size in ``(H, W)``.
    new_grid_size : tuple of int
        Target 3D grid size in ``(H, W, D)``.

    Returns
    -------
    torch.Tensor
        Resized positional embedding of shape ``(1, N_new, C)``.

    Notes
    -----
    The 2D grid is resized in-plane and then repeated along depth.
    The repetition is averaged by depth to preserve scale.
    """
    prefix = pos_embed[:, :num_prefix_tokens]
    grid = pos_embed[:, num_prefix_tokens:]

    H0, W0 = old_grid_size
    H1, W1, D1 = new_grid_size
    C = grid.shape[-1]

    grid = grid.reshape(1, H0, W0, C).permute(0, 3, 1, 2)  # (1, C, H0, W0)
    grid = nn.functional.interpolate(
        grid,
        size=(H1, W1),
        mode="bicubic",
        align_corners=False,
    )
    grid = grid.unsqueeze(-1).repeat(1, 1, 1, 1, D1) / max(
        D1, 1
    )  # (1, C, H1, W1, D1)
    grid = grid.permute(0, 2, 3, 4, 1).reshape(1, H1 * W1 * D1, C)
    return torch.cat([prefix, grid], dim=1)


def checkpoint_filter_fn(
    state_dict: dict[str, Tensor],
    model: VisionTransformer3D,
) -> OrderedDict[str, Tensor]:
    """
    Adapt a checkpoint for `VisionTransformer3D`.

    Parameters
    ----------
    state_dict : dict of str to torch.Tensor
        Input checkpoint state dictionary. May come from a 2D ViT or from a
        compatible 3D ViT.
    model : VisionTransformer3D
        Target model instance.

    Returns
    -------
    collections.OrderedDict
        Filtered state dictionary suitable for loading.

    Notes
    -----
    This function handles the common cases:
    - inflate a 2D patch projection kernel into 3D
    - resize 2D absolute position embeddings into a 3D grid
    - drop classifier weights if shapes do not match
    """
    out = OrderedDict()
    model_sd = model.state_dict()

    for k, v in state_dict.items():
        if k not in model_sd:
            continue

        # Inflate Conv2d patch projection weights to Conv3d.
        if k == "patch_embed.proj.weight":
            target = model_sd[k]
            if v.ndim == 4 and target.ndim == 5:
                kd = target.shape[2]
                v = v.unsqueeze(2).repeat(1, 1, kd, 1, 1) / kd
            elif v.shape != target.shape:
                continue

        # Resize / inflate positional embeddings.
        elif k == "pos_embed":
            target = model_sd[k]
            if v.shape != target.shape:
                old_n = v.shape[1] - model.num_prefix_tokens
                old_hw = int(old_n**0.5)
                if old_hw * old_hw == old_n:
                    v = _resize_pos_embed_2d_to_3d(
                        v,
                        num_prefix_tokens=model.num_prefix_tokens,
                        old_grid_size=(old_hw, old_hw),
                        new_grid_size=model.grid_size,
                    )
                else:
                    continue

        # Drop incompatible classifier weights.
        elif k.startswith("head.") and v.shape != model_sd[k].shape:
            continue

        if v.shape == model_sd[k].shape:
            out[k] = v

    return out


def vit_small_patch16_128(
    **kwargs,
) -> VisionTransformer3D:
    """
    Build a small 3D ViT.

    Parameters
    ----------
    **kwargs
        Forwarded to :class:`VisionTransformer3D`.

    Returns
    -------
    VisionTransformer3D
        Model instance.
    """
    return VisionTransformer3D(
        img_size=128,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        **kwargs,
    )


def vit_base_patch16_128(
    **kwargs,
) -> VisionTransformer3D:
    """
    Build a base 3D ViT.

    Parameters
    ----------
    **kwargs
        Forwarded to :class:`VisionTransformer3D`.

    Returns
    -------
    VisionTransformer3D
        Model instance.
    """
    return VisionTransformer3D(
        img_size=128,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        **kwargs,
    )


def vit_large_patch16_128(
    **kwargs,
) -> VisionTransformer3D:
    """
    Build a large 3D ViT.

    Parameters
    ----------
    **kwargs
        Forwarded to :class:`VisionTransformer3D`.

    Returns
    -------
    VisionTransformer3D
        Model instance.
    """
    return VisionTransformer3D(
        img_size=128,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        **kwargs,
    )
