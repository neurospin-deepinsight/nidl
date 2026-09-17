from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F  # ruff: ignore[lowercase-imported-as-non-lowercase]
from torch import nn
from torch.optim import Optimizer

from nidl.backbones.volume.vit3d_moe import (
    Block,
    apply_masks,
    moe_bias_update,
    repeat_interleave_batch,
    trunc_normal_,
)
from nidl.estimators.base import BaseEstimator, TransformerMixin
from nidl.estimators.ssl.utils.encoder import build_encoder
from nidl.estimators.ssl.utils.momentum import (
    MomentumUpdater,
    initialize_momentum_params,
)
from nidl.estimators.ssl.utils.optimizer import configure_ssl_optimizers
from nidl.utils.data_parsing import parse_x_or_xy_batch


class NeuroJEPAEncoderWrapper(nn.Module):
    """Thin interface-checking wrapper around a user-supplied 3D encoder.

    Parameters
    ----------
    vit : nn.Module
        Must expose ``embed_dim`` (int), ``patch_size`` (3-tuple),
        ``grid_shape`` (3-tuple), ``blocks`` (nn.ModuleList), and
        ``forward(x, masks=None) -> (tokens, moe_scores)``.
        :class:`~nidl.backbones.volume.VisionTransformer3DMoE` is a
        reference implementation satisfying this contract.

    Raises
    ------
    TypeError
        If ``vit`` is missing any of the required attributes.
    """

    _REQUIRED = ("embed_dim", "patch_size", "grid_shape", "blocks")

    def __init__(self, vit: nn.Module):
        super().__init__()
        missing = [a for a in self._REQUIRED if not hasattr(vit, a)]
        if missing:
            raise TypeError(
                "encoder must follow the NeuroJEPA 3D-ViT interface, "
                f"missing {missing}."
            )
        if not callable(getattr(vit, "forward", None)):
            raise TypeError("encoder must be callable (implement forward).")
        self.vit = vit

    @property
    def embed_dim(self) -> int:
        return self.vit.embed_dim

    @property
    def patch_size(self) -> tuple[int, int, int]:
        return self.vit.patch_size

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return self.vit.grid_shape

    @property
    def blocks(self) -> nn.ModuleList:
        return self.vit.blocks

    def forward(
        self, x: torch.Tensor, masks: Optional[list[torch.Tensor]] = None
    ):
        return self.vit(x, masks=masks)


class VisionTransformerPredictor3D(nn.Module):
    """Takes context-encoder tokens + (context indices, target indices) and
    predicts the target-encoder's latents at the target positions.

    Ported from: src/neurojepa/models/predictor.py
    """

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        embed_dim: int = 768,
        predictor_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_mask_tokens: int = 1,
        init_std: float = 0.02,
    ):
        super().__init__()
        grid_h, grid_w, grid_d = grid_shape
        self.grid_size = grid_h
        self.grid_depth = grid_d
        self.num_patches = grid_h * grid_w * grid_d

        self.predictor_embed = nn.Linear(
            embed_dim, predictor_embed_dim, bias=True
        )
        self.num_mask_tokens = num_mask_tokens
        self.mask_tokens = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
                for _ in range(num_mask_tokens)
            ]
        )

        self.predictor_pos_embed = None
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, predictor_embed_dim),
            requires_grad=False,
        )

        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    grid_size=self.grid_size,
                    use_moe=False,
                )
                for _ in range(depth)
            ]
        )
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj = nn.Linear(
            predictor_embed_dim, embed_dim, bias=True
        )

        self.init_std = init_std
        for mt in self.mask_tokens:
            trunc_normal_(mt, std=init_std)
        self.apply(self._init_weights)
        for layer_id, blk in enumerate(self.predictor_blocks):
            blk.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            blk.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        masks_x: list[torch.Tensor],
        masks_y: list[torch.Tensor],
        mask_index: int = 0,
    ):
        """
        x : (B * len(masks_x), K_ctx, E) -- context-encoder output.
        masks_x : the same list of context-index tensors used to produce `x`.
        masks_y : list of (B, K_pred) target-index tensors to predict.

        Returns (B * len(masks_x) * len(masks_y), K_pred, E). In practice
        `NeuroJEPA._shared_step` always calls this with single-element
        `masks_x`/`masks_y` lists (one mask "scale" per call), so the
        general cross-product machinery below collapses to a plain,
        nidl-style single-context/single-target forward; it is kept general
        so multiple target blocks per context (à la vanilla I-JEPA) also work.
        """
        B = len(x) // len(masks_x)
        x = self.predictor_embed(x)

        _, N_ctxt, _ = x.shape

        mask_index = mask_index % self.num_mask_tokens
        pred_tok = self.mask_tokens[mask_index].repeat(B, self.num_patches, 1)
        pred_tok = apply_masks(
            pred_tok, masks_y
        )  # (B*len(masks_y), K_pred, E)
        pred_tok = repeat_interleave_batch(
            pred_tok, B, repeat=len(masks_x)
        )  # row order matches x.repeat below

        x = x.repeat(len(masks_y), 1, 1)
        x = torch.cat([x, pred_tok], dim=1)

        mx = torch.cat(masks_x, dim=0).repeat(len(masks_y), 1)
        my = repeat_interleave_batch(
            torch.cat(masks_y, dim=0), B, repeat=len(masks_x)
        )
        pos_idx = torch.cat([mx, my], dim=1)

        for blk in self.predictor_blocks:
            x, _ = blk(
                x,
                mask=pos_idx,
                D_patches=self.grid_depth,
                H_patches=self.grid_size,
                W_patches=self.grid_size,
            )
        x = self.predictor_norm(x)
        x = x[:, N_ctxt:]
        return self.predictor_proj(x)


def compute_foreground_patches(
    volumes: torch.Tensor,
    patch_size,
    threshold: float = 0.0,
    min_foreground_fraction: float = 0.1,
) -> torch.Tensor:
    """Per-patch boolean foreground map from voxel intensities: a patch
    counts as foreground if at least `min_foreground_fraction` of its voxels
    exceed a per-sample, data-driven intensity threshold.

    Ported from `src/neurojepa/masks/masking.py::compute_foreground_patches`.
    """
    pH, pW, pD = patch_size
    B = volumes.shape[0]
    vol = volumes.amax(dim=1)  # (B, H, W, D), collapse channel dim

    vol_flat = vol.reshape(B, -1)
    stride = max(1, vol_flat.shape[1] // 100_000)
    sub = vol_flat[:, ::stride].float()
    p2 = torch.quantile(sub, 0.02, dim=1)
    p98 = torch.quantile(sub, 0.98, dim=1)
    dyn_thresh = torch.where(
        p98 > p2, 0.1 * (p98 - p2) + p2, torch.full_like(p2, threshold)
    )
    dyn_thresh = dyn_thresh.view(B, 1, 1, 1)

    voxel_mask = (vol > dyn_thresh).float().unsqueeze(1)
    fg_frac = F.avg_pool3d(
        voxel_mask, kernel_size=(pH, pW, pD), stride=(pH, pW, pD)
    ).squeeze(1)
    return fg_frac >= min_foreground_fraction  # (B, nH, nW, nD)


@dataclass
class MaskScaleConfig:
    """One entry of the `mask:` list in `pretrain_neurojepa_base.yaml`."""

    spatial_scale: tuple[float, float] = (0.0, 0.2)
    depth_scale: tuple[float, float] = (0.0, 1.0)
    aspect_ratio: tuple[float, float] = (0.75, 1.5)
    num_blocks: int = 32
    total_mask_ratio: float = 0.75


# The exact 3 scales used in the paper's released base config.
DEFAULT_MULTISCALE_MASK_CONFIG: tuple[MaskScaleConfig, ...] = (
    MaskScaleConfig(
        spatial_scale=(0.0, 0.2), num_blocks=32, total_mask_ratio=0.75
    ),
    MaskScaleConfig(
        spatial_scale=(0.2, 0.5), num_blocks=16, total_mask_ratio=0.75
    ),
    MaskScaleConfig(
        spatial_scale=(0.5, 0.7), num_blocks=4, total_mask_ratio=0.75
    ),
)


class _MaskGenerator:
    """Generates ONE complete (context, target) split of the patch grid by
    carving `num_blocks` axis-aligned blocks out of the grid, unioning them
    into a single masked region, then adjusting until it hits exactly
    `total_mask_ratio` of all patches. One "scale" in the multi-scale scheme;
    `MultiScaleMaskCollator` runs several of these (one per `MaskScaleConfig`)
    per training step.

    Ported from: src/neurojepa/masks/masking.py::_MaskGenerator
    """

    def __init__(self, grid_shape: tuple[int, int, int], cfg: MaskScaleConfig):
        self.height, self.width, self.depth = grid_shape
        self.cfg = cfg
        self.num_patches = self.height * self.width * self.depth
        self.target_pred = int(self.num_patches * cfg.total_mask_ratio)
        self.target_enc = self.num_patches - self.target_pred
        assert self.target_enc > 0, "total_mask_ratio too high for this grid"

    def _sample_block_size(self, g: torch.Generator):
        cfg = self.cfg
        d_scale = cfg.depth_scale[0] + torch.rand(1, generator=g).item() * (
            cfg.depth_scale[1] - cfg.depth_scale[0]
        )
        d = max(1, min(int(self.depth * d_scale), self.depth))

        s_scale = cfg.spatial_scale[0] + torch.rand(1, generator=g).item() * (
            cfg.spatial_scale[1] - cfg.spatial_scale[0]
        )
        area = max(1, int(self.height * self.width * s_scale))

        min_ar, max_ar = cfg.aspect_ratio
        log_ar = math.log(min_ar) + torch.rand(1, generator=g).item() * (
            math.log(max_ar) - math.log(min_ar)
        )
        ar = math.exp(log_ar)

        h = max(1, min(round(math.sqrt(area * ar)), self.height))
        w = max(1, min(round(math.sqrt(area / ar)), self.width))
        return d, h, w

    def __call__(
        self,
        batch_size: int,
        seed: int,
        foreground_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (enc_idx, pred_idx), each a (B, K) LongTensor."""
        enc_list, pred_list = [], []
        for si in range(batch_size):
            g = torch.Generator()
            g.manual_seed(seed * 1_000_003 + si)

            mask = torch.ones(
                self.height, self.width, self.depth, dtype=torch.bool
            )
            for _ in range(self.cfg.num_blocks):
                d, h, w = self._sample_block_size(g)
                top = torch.randint(
                    0, max(1, self.height - h + 1), (1,), generator=g
                ).item()
                left = torch.randint(
                    0, max(1, self.width - w + 1), (1,), generator=g
                ).item()
                start = torch.randint(
                    0, max(1, self.depth - d + 1), (1,), generator=g
                ).item()
                mask[top : top + h, left : left + w, start : start + d] = (
                    False  # carved out => target
                )

            fg_flat = (
                foreground_mask[si].flatten().bool()
                if foreground_mask is not None
                else None
            )
            mask = self._adjust_to_target(mask, g, fg_flat)

            mask_flat = mask.flatten()
            enc_idx = mask_flat.nonzero(as_tuple=True)[0]
            pred_idx = (~mask_flat).nonzero(as_tuple=True)[0]
            enc_idx = enc_idx[torch.randperm(len(enc_idx), generator=g)]
            pred_idx = pred_idx[torch.randperm(len(pred_idx), generator=g)]
            enc_list.append(enc_idx)
            pred_list.append(pred_idx)

        return torch.stack(enc_list), torch.stack(pred_list)

    def _adjust_to_target(
        self,
        mask: torch.Tensor,
        g: torch.Generator,
        fg_flat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Randomly flip kept<->masked patches until
        `mask.sum() == target_enc`, biased (when a foreground map is available)
        to erode background first / restore foreground first. Simplified vs.
        the official boundary-erosion loop (which grows/shrinks along the mask
        boundary for spatial coherence)."""
        mask_flat = mask.flatten()
        n_keep = mask_flat.sum().item()

        if n_keep > self.target_enc:
            kept = mask_flat.nonzero(as_tuple=True)[0]
            if fg_flat is not None:
                bg = kept[~fg_flat[kept]]
                fg = kept[fg_flat[kept]]
                order = torch.cat(
                    [
                        bg[torch.randperm(len(bg), generator=g)],
                        fg[torch.randperm(len(fg), generator=g)],
                    ]
                )
            else:
                order = kept[torch.randperm(len(kept), generator=g)]
            mask_flat[order[: n_keep - self.target_enc]] = False
        elif n_keep < self.target_enc:
            masked = (~mask_flat).nonzero(as_tuple=True)[0]
            if fg_flat is not None:
                fg = masked[fg_flat[masked]]
                bg = masked[~fg_flat[masked]]
                order = torch.cat(
                    [
                        fg[torch.randperm(len(fg), generator=g)],
                        bg[torch.randperm(len(bg), generator=g)],
                    ]
                )
            else:
                order = masked[torch.randperm(len(masked), generator=g)]
            mask_flat[order[: self.target_enc - n_keep]] = True

        return mask_flat.view(self.height, self.width, self.depth)


class MultiScaleMaskCollator:
    """Runs one `_MaskGenerator` per `MaskScaleConfig`, plus (optionally) the
    foreground computation.

    Ported from: src/neurojepa/masks/masking.py::MaskCollator
    """

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        scale_configs: Sequence[
            MaskScaleConfig
        ] = DEFAULT_MULTISCALE_MASK_CONFIG,
        foreground_aware: bool = True,
        foreground_threshold: float = 0.0,
        min_foreground_fraction: float = 0.1,
    ):
        self.patch_size = patch_size
        self.foreground_aware = foreground_aware
        self.foreground_threshold = foreground_threshold
        self.min_foreground_fraction = min_foreground_fraction
        self.generators = [
            _MaskGenerator(grid_shape, cfg) for cfg in scale_configs
        ]
        self._step_counter = -1
        self._rank_seed_offset = 0

    def set_rank(self, rank: int, large_multiplier: int = 10_000_000) -> None:
        """Call once per DDP process (see `NeuroJEPA.on_fit_start`) so
        different ranks draw different mask geometries at the same global
        step."""
        self._rank_seed_offset = rank * large_multiplier

    def step(self) -> int:
        self._step_counter += 1
        return self._step_counter + self._rank_seed_offset

    def __call__(self, volumes: torch.Tensor):
        """
        volumes : (B, C, H, W, D) tensor for THIS step.

        Returns
        -------
        masks_enc  : list of (B, K_enc) LongTensors, one per scale config.
        masks_pred : list of (B, K_pred) LongTensors, one per scale config.
        fg_flat    : (B, N) float tensor of foreground fraction per patch,
                     or None if foreground_aware=False.
        """
        B = volumes.shape[0]
        fg_mask = None
        if self.foreground_aware:
            fg_mask = compute_foreground_patches(
                volumes,
                self.patch_size,
                self.foreground_threshold,
                self.min_foreground_fraction,
            )
            fg_mask = fg_mask.cpu()

        seed = self.step()
        masks_enc, masks_pred = [], []
        for gen in self.generators:
            enc, pred = gen(B, seed, foreground_mask=fg_mask)
            masks_enc.append(enc)
            masks_pred.append(pred)

        fg_flat = fg_mask.flatten(1).float() if fg_mask is not None else None
        return masks_enc, masks_pred, fg_flat


def foreground_aware_jepa_loss(
    z: list[torch.Tensor],
    h: list[torch.Tensor],
    masks_pred: list[torch.Tensor],
    loss_exp: float = 1.0,
    fg_map: Optional[torch.Tensor] = None,
    bg_weight: float = 1.0,
) -> torch.Tensor:
    """
    z : list (length = num_scales), one (B, K_i, D) predictor-output tensor
        per mask "scale" (`NeuroJEPA._shared_step` calls the encoder +
        predictor once per scale, matched context/target pairs).
    h : same structure as z, target-encoder latents.
    masks_pred : list (length = num_scales) of (B, K) index tensors, used to
        gather the matching per-patch foreground weights out of `fg_map`.
    fg_map : (B, N) foreground fraction per patch, or None to disable
        weighting (falls back to a uniform mean L1, i.e. nidl's plain loss).
    bg_weight : weight assigned to a pure-background patch (fg=0); a pure
        foreground patch (fg=1) always gets weight 1.0.

    Ported from: src/neurojepa/loss/jepa_loss.py
    """
    use_weighting = fg_map is not None and bg_weight < 1.0
    pred_fg = None
    if use_weighting:
        pred_fg = [
            apply_masks(fg_map.unsqueeze(-1), [mi], concat=False)[0]
            for mi in masks_pred
        ]

    loss, n = 0.0, 0
    for scale_idx, (zi, hi) in enumerate(zip(z, h)):
        per_token = torch.abs(zi - hi) ** loss_exp / loss_exp  # (B, K, D)
        per_token = per_token.mean(dim=-1)  # (B, K)

        if use_weighting:
            fg_vals = pred_fg[scale_idx].squeeze(-1)  # (B, K)
            w = bg_weight + (1.0 - bg_weight) * fg_vals
            w = w / w.mean(dim=-1, keepdim=True).clamp(min=1e-6)
            loss = loss + (per_token * w).mean()
        else:
            loss = loss + per_token.mean()
        n += 1
    return loss / n


class NeuroJEPA(TransformerMixin, BaseEstimator):
    """Implementation of Neuro-JEPA [1]_.

    This solver predicts the representations of missing parts
    of the input (here: 3D brain MRI patches) from a visible context, using
    a context encoder, an EMA target encoder, and a predictor.

    Compared to I-JEPA (3d), it uses:

    - multi-scale block masking (several differently-shaped-but-equal-ratio
      maskings per volume instead of one),
    - a sparse Mixture-of-Experts backbone (via the `encoder` you pass in),
    - a foreground-aware loss that down-weights background (non-brain)
      voxel-patches.

    Parameters
    ----------
    encoder : nn.Module
        3D ViT-like encoder. Must expose ``embed_dim``, ``patch_size``,
        ``grid_shape``, ``blocks``, and ``forward(x, masks=None)``. See
        :class:`~nidl.backbones.volume.VisionTransformer3DMoE` for a
        reference implementation (with or without a sparse MoE backbone --
        pass ``use_moe=True`` to that constructor and set ``use_moe=True``
        here too so the MoE bias update runs during training).

    mask_scale_configs : sequence of MaskScaleConfig, \
        default=3-scale config from [1]_
        One entry per masking "scale": each draws `num_blocks` blocks with
        sizes controlled by `spatial_scale`/`depth_scale`/`aspect_ratio`,
        unions them, and adjusts to hit exactly `total_mask_ratio` of all
        patches. By default, a small-block (32 blocks, 0-20% spatial
        scale), a medium-block (16 blocks, 20-50% spatial scale), and a
        large-block (4 blocks, 50-70% spatial scale) masking are all drawn
        for every volume, at every step, each targeting a 75% mask ratio.

    foreground_aware : bool, default=True
        Whether to compute a per-patch foreground map (voxel-intensity based)
        and use it to (a) bias mask erosion/dilation toward removing
        background patches first, and (b) down-weight background patches in
        the loss (see `bg_weight`).

    foreground_threshold : float, default=0.0
        Fallback per-sample voxel-intensity threshold used by
        `compute_foreground_patches` when the sample's data-driven
        (2nd/98th percentile based) threshold cannot be estimated.

    min_foreground_fraction : float, default=0.1
        Minimum fraction of foreground voxels a patch must contain to be
        itself counted as a foreground patch (see
        `compute_foreground_patches`).

    loss_exp : float, default=1.0
        Exponent of the per-token L1-style loss (`|pred - target|^p / p`).

    bg_weight : float, default=0.1
        Loss weight for a pure-background patch; a pure-foreground patch
        always has weight 1.0. Set to 1.0 to disable foreground weighting
        (recovers a uniform loss, i.e. `IJEPA`'s `SmoothL1Loss` in spirit).

    use_moe : bool, default=False
        Whether `encoder.blocks` contains sparse MoE layers that need the
        post-step, aux-loss-free bias update (see
        `vision_transformer_3d.moe_bias_update`). Set to match however you
        built `encoder`.

    moe_bias_update_rate : float, default=1e-4
        Step size of the MoE router bias update (ignored if `use_moe=False`).

    moe_bias_clip : float, default=0.3
        Maximum absolute value of the MoE router bias after each update
        (ignored if `use_moe=False`).

    predictor_embed_dim : int, default=384
        Dimension of the predictor hidden layers. It can be different from
        the encoder output dimension.

    predictor_depth : int, default=6
        Number of Transformer blocks in the predictor.

    predictor_num_heads : int, default=12
        Number of attention heads in the predictor.

    optimizer : {'sgd', 'adam', 'adamW'} or Optimizer, default='adamW'
        Optimizer for training the model. If a string is given, it can be:

            - 'sgd': Stochastic Gradient Descent (with optional momentum).
            - 'adam': First-order gradient-based optimizer.
            - 'adamW' (default): Adam with decoupled weight decay
              regularization (see "Decoupled Weight Decay Regularization",
              Loshchilov and Hutter, ICLR 2019).

    learning_rate : float, default=6e-4
        Initial learning rate.

    weight_decay : float, default=0.04
        Weight decay in the optimizer.

    exclude_bias_and_norm_wd : bool, default=True
        Whether the bias terms and normalization layers get weight decay
        during optimization or not.

    ema_start : float, default=0.99925
        Base value for the weighting coefficient in the target encoder
        momentum update with exponential moving average. A cosine
        annealing scheme is used.

    ema_end : float, default=1.0
        Final value for the weighting coefficient in the target encoder
        momentum update.

    optimizer_kwargs : dict or None, default=None
        Extra named arguments for the optimizer.

    lr_scheduler : {"none", "warmup_cosine"}, LRSchedulerPLType or None, \
        default="warmup_cosine"
        Learning rate scheduler to use.

    lr_scheduler_kwargs : dict or None, default=None
        Extra named arguments for the scheduler. By default, it is set to
        {"warmup_epochs": 10, "warmup_start_lr": 1e-6, "min_lr": 0.0,
        "interval": "step"}.

    **kwargs : dict, optional
        Extra named arguments for the `BaseEstimator` class (given to the
        PL `Trainer`), such as `max_epochs`, `max_steps`, `callbacks`, etc.

    Attributes
    ----------
    context_encoder : NeuroJEPAEncoderWrapper
        Wraps the trainable copy of `encoder`. Used at inference (`transform`).
    target_encoder : NeuroJEPAEncoderWrapper
        Wraps the EMA copy of `encoder`; not used at inference.
    predictor : VisionTransformerPredictor3D
        Predicts target-encoder latents from context-encoder latents.
    masker : MultiScaleMaskCollator
        Generates the per-step multi-scale (context, target) mask pairs.

    References
    ----------
    .. [1] Huang et al., "Learning Sparse Latent Predictive Foundation
           Model for Multimodal Neuroimaging", arXiv:2606.14957.
    """

    def __init__(
        self,
        encoder: nn.Module,
        mask_scale_configs: Sequence[
            MaskScaleConfig
        ] = DEFAULT_MULTISCALE_MASK_CONFIG,
        foreground_aware: bool = True,
        foreground_threshold: float = 0.0,
        min_foreground_fraction: float = 0.1,
        loss_exp: float = 1.0,
        bg_weight: float = 0.1,
        use_moe: bool = False,
        moe_bias_update_rate: float = 1e-4,
        moe_bias_clip: float = 0.3,
        predictor_embed_dim: int = 384,
        predictor_depth: int = 6,
        predictor_num_heads: int = 12,
        optimizer: Union[str, Optimizer] = "adamW",
        learning_rate: float = 6e-4,
        weight_decay: float = 0.04,
        exclude_bias_and_norm_wd: bool = True,
        ema_start: float = 0.99925,
        ema_end: float = 1.0,
        optimizer_kwargs: Optional[dict] = None,
        lr_scheduler: Optional[Union[str, Any]] = "warmup_cosine",
        lr_scheduler_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ):
        ignore = kwargs.pop("ignore", ["callbacks", "encoder"])
        if "callbacks" not in ignore:
            ignore.append("callbacks")
        if "encoder" not in ignore:
            ignore.append("encoder")

        super().__init__(**kwargs, ignore=ignore)

        self.parse_batch = parse_x_or_xy_batch

        # -- independent context / target encoder copies, exactly like IJEPA
        self.target_encoder = NeuroJEPAEncoderWrapper(
            build_encoder(encoder, deepcopy=True)
        )
        self.context_encoder = NeuroJEPAEncoderWrapper(build_encoder(encoder))
        initialize_momentum_params(self.context_encoder, self.target_encoder)
        self.momentum_updater = MomentumUpdater(ema_start, ema_end)

        grid_shape = self.target_encoder.grid_shape
        embed_dim = self.target_encoder.embed_dim
        patch_size = self.target_encoder.patch_size

        self.predictor = VisionTransformerPredictor3D(
            grid_shape=grid_shape,
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
        )

        self.masker = MultiScaleMaskCollator(
            grid_shape=grid_shape,
            patch_size=patch_size,
            scale_configs=mask_scale_configs,
            foreground_aware=foreground_aware,
            foreground_threshold=foreground_threshold,
            min_foreground_fraction=min_foreground_fraction,
        )

        self.foreground_aware = foreground_aware
        self.loss_exp = loss_exp
        self.bg_weight = bg_weight
        self.use_moe = use_moe
        self.moe_bias_update_rate = moe_bias_update_rate
        self.moe_bias_clip = moe_bias_clip

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.exclude_bias_and_norm_wd = exclude_bias_and_norm_wd
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = (
            lr_scheduler_kwargs if lr_scheduler_kwargs is not None else {}
        )
        self.optimizer_kwargs = optimizer_kwargs

        self._fill_default_lr_scheduler_kwargs()

    def _shared_step(self, batch):
        x, _ = self.parse_batch(batch, device=self.device)

        masks_enc, masks_pred, fg_flat = self.masker(x)
        masks_enc = [m.to(x.device, non_blocking=True) for m in masks_enc]
        masks_pred = [m.to(x.device, non_blocking=True) for m in masks_pred]
        if fg_flat is not None:
            fg_flat = fg_flat.to(x.device, non_blocking=True)

        num_scales = len(masks_enc)

        # The (expensive) full-volume EMA target pass happens ONCE and is
        # sliced into per-scale target tokens afterwards.
        h_scales = self.forward_target(
            x, masks_pred
        )  # list[num_scales] of (B, K_pred_i, D)

        # Context encoder + predictor run ONCE PER SCALE with a single
        # matched (context-mask, target-mask) pair -- i.e. what "multi-scale
        # masking" means concretely: 3 independent JEPA predictions per
        # volume, one per mask geometry, losses summed below.
        z_scales = []
        for i in range(num_scales):
            zi = self.context_encoder(x, masks=[masks_enc[i]])[
                0
            ]  # (B, K_enc_i, D)
            zi = self.predictor(
                zi, masks_x=[masks_enc[i]], masks_y=[masks_pred[i]]
            )  # (B, K_pred_i, D)
            z_scales.append(zi)

        loss = foreground_aware_jepa_loss(
            z_scales,
            h_scales,
            masks_pred,
            loss_exp=self.loss_exp,
            fg_map=fg_flat,
            bg_weight=self.bg_weight,
        )
        return {
            "loss": loss,
            "z_pred": z_scales,
            "z_target": h_scales,
        }

    def training_step(self, batch, batch_idx: int):
        """Perform one training step and compute the training loss.

        Parameters
        ----------
        batch : torch.Tensor or (torch.Tensor, torch.Tensor)
            ``X`` or ``(X, Y)`` where ``X`` has shape ``(B, C, H, W, D)``.
            ``Y`` (eventual labels) is ignored.
        batch_idx : int
            Ignored.

        Returns
        -------
        outputs : dict
            ``"loss"``, ``"z_pred"`` (list[num_scales] of predictor outputs),
            ``"z_target"`` (list[num_scales] of target-encoder outputs).
        """
        outputs = self._shared_step(batch)
        self.log("loss/train", outputs["loss"], prog_bar=True, sync_dist=True)
        return outputs

    @torch.no_grad()
    def forward_target(
        self, x: torch.Tensor, masks_pred: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Full (unmasked) forward through the EMA target encoder,
        layer-normed, then sliced into per-scale target latents at the
        masked positions. Mirrors `IJEPA.forward_target`."""
        h = self.target_encoder(x, masks=None)[0]
        h = F.layer_norm(h, (h.size(-1),))
        return apply_masks(h, masks_pred, concat=False)

    def on_train_batch_end(self, outputs: dict, batch, batch_idx: int):
        """Performs the teacher momentum update (and, if `use_moe=True`, the
        MoE aux-loss-free bias update), matching the official
        `train_one_epoch`'s post-step ordering: MoE bias update, then EMA.

        Parameters
        ----------
        outputs : dict
            Outputs of the training step (ignored).
        batch : torch.Tensor or pair of torch.Tensor
            Ignored.
        batch_idx : int
            Ignored.
        """
        if self.use_moe:
            minvio, maxvio = moe_bias_update(
                self.context_encoder,
                self.moe_bias_update_rate,
                self.moe_bias_clip,
            )
            self.log("moe/min_violation", minvio)
            self.log("moe/max_violation", maxvio)

        self.momentum_updater.update(self.context_encoder, self.target_encoder)
        self.log("lambda", self.momentum_updater.cur_lambda)
        self.momentum_updater.update_lambda(
            cur_step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
        )

    def on_fit_start(self):
        """Decorrelate mask geometry across DDP ranks.

        `self.trainer.global_rank` only exists once a Trainer/strategy is
        attached (i.e. not yet at `__init__` time, when the estimator is
        merely constructed) -- `on_fit_start` is the first hook guaranteed
        to run after that setup and before the first training step, so it's
        the right place to set it. Without this, every rank's `self.masker`
        starts its own step counter at 0 and increments in lockstep with
        every other rank, with nothing else rank-dependent in the seed --
        so all ranks would draw the identical mask geometry for the sample
        at a given local batch index every step, even though the actual
        volume there differs per rank. Single-GPU / single-process runs are
        unaffected (`global_rank == 0`, offset is a no-op).
        """
        self.masker.set_rank(self.trainer.global_rank)

    def validation_step(self, batch: Any, batch_idx: int):
        """Performs one validation step and computes the validation loss.
        Same return structure as `training_step`."""
        outputs = self._shared_step(batch)
        self.log("loss/val", outputs["loss"], prog_bar=True, sync_dist=True)
        return outputs

    def test_step(self, batch, batch_idx):
        """Skip the test step."""
        return

    def transform_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        """Encode the input data into the latent space (no masking, no
        predictor -- the predictor is only used during training).

        Parameters
        ----------
        batch : torch.Tensor
            (B, C, H, W, D) volume batch, given as-is to the context encoder.
        batch_idx, dataloader_idx : int
            Ignored.

        Returns
        -------
        features : torch.Tensor
            Context-encoder features averaged across the token dimension,
            shape (B, embed_dim).
        """
        tokens, _ = self.context_encoder(batch, masks=None)
        return tokens.mean(dim=1)

    def configure_optimizers(self):
        """Initialize the optimizer and learning rate scheduler."""
        params = [
            {"name": "backbone", "params": self.context_encoder.parameters()},
            {"name": "predictor", "params": self.predictor.parameters()},
        ]
        return configure_ssl_optimizers(
            trainer=self.trainer,
            optim_params=params,
            optimizer=self.optimizer,
            optimizer_kwargs=self.optimizer_kwargs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            exclude_bias_and_norm_wd=self.exclude_bias_and_norm_wd,
            lr_scheduler=self.lr_scheduler,
            lr_scheduler_kwargs=self.lr_scheduler_kwargs,
        )

    def _fill_default_lr_scheduler_kwargs(self):
        if self.lr_scheduler_kwargs is None:
            self.lr_scheduler_kwargs = {}
        self.lr_scheduler_kwargs.setdefault("warmup_epochs", 10)
        self.lr_scheduler_kwargs.setdefault("interval", "step")
        self.lr_scheduler_kwargs.setdefault("warmup_start_lr", 1e-6)
        self.lr_scheduler_kwargs.setdefault("min_lr", 0)
