from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F  # ruff: ignore[lowercase-imported-as-non-lowercase]
from torch import distributed as dist
from torch import nn


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    """Truncated-normal init, identical to the official `utils/tensors.py`."""

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def apply_masks(
    x: torch.Tensor, masks: Sequence[torch.Tensor], concat: bool = True
):
    """Gathers, per mask in `masks`, the tokens of `x` indexed by that mask,
    then concatenates the results along the batch dimension. Verbatim (up to
    formatting) `src/neurojepa/masks/utils.py`.

    Parameters
    ----------
    x : (B, N, D) tensor of tokens.
    masks : list of (B, K) LongTensors, each containing indices in [0, N)
        of the tokens to keep for that particular mask.

    Returns
    -------
    (B * len(masks), K, D) tensor if concat, else a list of (B, K, D) tensors.
    """
    all_x = []
    for m in masks:
        batch_idx = torch.arange(x.size(0), device=x.device).unsqueeze(1)
        all_x.append(x[batch_idx, m])
    if not concat:
        return all_x
    return torch.cat(all_x, dim=0)


def repeat_interleave_batch(
    x: torch.Tensor, B: int, repeat: int
) -> torch.Tensor:
    """[g0_0..g0_{B-1}, g1_0..g1_{B-1}, ...] (N=len(x)//B groups of size B)
    -> each group repeated `repeat` times, concatenated group-by-group.
    Verbatim `src/neurojepa/utils/tensors.py`."""
    N = len(x) // B
    return torch.cat(
        [
            torch.cat([x[i * B : (i + 1) * B] for _ in range(repeat)], dim=0)
            for i in range(N)
        ],
        dim=0,
    )


class PatchEmbed3D(nn.Module):
    """Tokenizes a (B, C, H, W, D) volume into non-overlapping 3D patches
    ("tubelets") with a single strided Conv3d -- this is what makes I-JEPA
    (and V-JEPA2) trivially extensible to 3D: only this module and the mask
    geometry need to become 3D-aware."""

    def __init__(
        self, patch_size=(12, 12, 12), in_chans: int = 1, embed_dim: int = 768
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


# ==========================================================================
# Mixture-of-Experts
#   Aux-loss-free, bias-corrected top-k router (DeepSeek-V3 style) with a
#   handful of *shared* experts (always active, dense) plus N *routed*
#   experts (only top-k active per token). Lets different experts
#   specialize by anatomical region / tissue type instead of
#   every token going through one shared dense MLP.
# ==========================================================================


class MLP(nn.Module):
    """A single expert (or the dense fallback MLP when MoE is off)."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Gate(nn.Module):
    """Router: scores every token against every routed expert, keeps the
    top-k, and maintains a per-expert *bias* (not a gradient-based auxiliary
    load-balancing loss) that is nudged after every step by
    `moe_bias_update` below so that under-used experts become more likely to
    be picked next step -- the "aux-loss-free" trick from DeepSeek-V3,
    reused as-is by Neuro-JEPA."""

    def __init__(
        self,
        dim: int,
        n_routed_experts: int,
        n_activated_experts: int,
        score_func: str = "softmax",
        route_scale: float = 1.0,
    ):
        super().__init__()
        self.topk = n_activated_experts
        self.score_func = score_func
        self.gate = nn.Linear(dim, n_routed_experts, bias=False)
        self.register_buffer("bias", torch.zeros(n_routed_experts))
        self.route_scale = route_scale

    def forward(self, x: torch.Tensor):
        logits = self.gate(x)
        scores = (
            logits.softmax(dim=-1)
            if self.score_func == "softmax"
            else logits.sigmoid()
        )
        original_scores = scores
        scores = (
            scores + self.bias
        )  # bias only steers *selection*, not the weight values
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(
                min=1e-9
            )
        weights = weights * self.route_scale
        return weights.type_as(x), indices, original_scores


class MoE(nn.Module):
    """Sparse feed-forward block: `n_shared_experts` dense experts (always
    on, summed) + `n_routed_experts` experts of which only the top
    `n_activated_experts` fire per token."""

    def __init__(
        self,
        dim: int,
        n_shared_experts: int,
        n_routed_experts: int,
        n_activated_experts: int,
        moe_inter_dim: int,
        score_func: str = "softmax",
        route_scale: float = 1.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.gate = Gate(
            dim, n_routed_experts, n_activated_experts, score_func, route_scale
        )
        self.experts = nn.ModuleList(
            [
                MLP(dim, moe_inter_dim, act_layer=act_layer, drop=drop)
                for _ in range(n_routed_experts)
            ]
        )
        self.shared_experts = MLP(
            dim,
            n_shared_experts * moe_inter_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.register_buffer(
            "counts", torch.zeros(n_routed_experts, dtype=torch.long)
        )

    def forward(self, x: torch.Tensor):
        shape = x.size()
        x = x.reshape(-1, self.dim)
        weights, indices, scores = self.gate(x)

        self.counts += torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts
        ).to(self.counts.device)

        y = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            idx, top = torch.where(indices == i)
            if idx.numel() == 0:
                y = (
                    y + 0.0 * expert(x[:1]).sum()
                )  # keep expert in autograd graph
                continue
            y[idx] += expert(x[idx]) * weights[idx, top, None]

        z = self.shared_experts(x)
        return (y + z).view(shape), scores

    def reset_counts(self):
        self.counts.zero_()


def moe_bias_update(
    model: nn.Module, update_rate: float, bias_clip: float = 0.3
) -> tuple[float, float]:
    """Aux-loss-free load-balancing update, ported from
    `models/utils/moe.py::moe_bias_update`.

    `model` must expose `.blocks` (a ModuleList of `Block` instances, some
    of which may hold a `MoE` in `.mlp`). Call this once per optimizer step,
    AFTER `optimizer.step()`. Returns (min, max) expert count relative to
    the average count, for logging routing balance.
    """
    moe_layers = [b.mlp for b in model.blocks if isinstance(b.mlp, MoE)]
    if not moe_layers:
        return 1.0, 1.0
    distributed = dist.is_available() and dist.is_initialized()
    total_min, total_max = 0.0, 0.0
    with torch.no_grad():
        for moe in moe_layers:
            counts = moe.counts.float()
            if distributed:
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            avg = counts.mean()
            if avg > 0:
                rel = counts / avg
                total_min += rel.min().item()
                total_max += rel.max().item()
                error = avg - counts  # positive => under-used expert
                update = torch.sign(error)
                update = update - update.mean()  # keep bias zero-mean
                moe.gate.bias += update_rate * update
                moe.gate.bias.clamp_(-bias_clip, bias_clip)
                moe.gate.bias -= moe.gate.bias.mean()
            else:
                total_min += 1.0
                total_max += 1.0
            moe.reset_counts()
    n = len(moe_layers)
    return total_min / n, total_max / n


@dataclass
class MoEParams:
    """Hyperparameters of the sparse Mixture-of-Experts (MoE) layers used
    by `VisionTransformer3DMoE` when `use_moe=True`.

    Parameters
    ----------
    dim : int, default=768
        Token embedding dimension (must match the encoder's `embed_dim`).
    n_shared_experts : int, default=2
        Number of "shared" experts, always active for every token.
    n_routed_experts : int, default=16
        Number of routed experts to choose from.
    n_activated_experts : int, default=6
        Number of routed experts activated (top-k) per token.
    moe_inter_dim : int, default=384
        Hidden dimension of each expert's MLP.
    score_func : {"softmax", "sigmoid"}, default="softmax"
        Function used to turn router logits into routing scores.
    route_scale : float, default=4.0
        Scalar applied to the routing weights.
    bias_clip : float, default=0.3
        Passed to `moe_bias_update` as `bias_clip`.
    bias_update_rate : float, default=1e-4
        Passed to `moe_bias_update` as `update_rate`.
    moe_layer_indices : tuple of int, default=(1, 3, 5, 7, 9, 11)
        0-indexed block indices that use a sparse MoE instead of a dense
        MLP.
    """

    dim: int = 768
    n_shared_experts: int = 2
    n_routed_experts: int = 16
    n_activated_experts: int = 6
    moe_inter_dim: int = 384
    score_func: str = "softmax"
    route_scale: float = 4.0
    bias_clip: float = 0.3
    bias_update_rate: float = 1e-4
    # Layer indices (0-indexed) that use MoE; the rest use a plain dense MLP.
    # Matches the official base config: every other block from 1 to 11.
    moe_layer_indices: tuple[int, ...] = (1, 3, 5, 7, 9, 11)


def rotate_queries_or_keys(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Applies rotary position embedding to `x` at fractional positions
    `pos`."""
    _, _, _, D = x.size()
    assert D % 2 == 0
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega = 1.0 / (10000 ** (omega / (D / 2.0)))
    freq = torch.einsum("...,f->...f", pos, omega)
    emb_sin = freq.sin().repeat(1, 1, 1, 2)
    emb_cos = freq.cos().repeat(1, 1, 1, 2)
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return x * emb_cos + y * emb_sin


class RoPEAttention(nn.Module):
    """3D rotary-position self-attention: query/key channels are split into
    a depth-group / height-group / width-group and each group is rotated by
    the token's position along that axis. It lets the encoder
    work on 3D volumes without any additive absolute positional embedding.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        grid_size=16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop_p = attn_drop
        self.d_dim = self.h_dim = self.w_dim = int(
            2 * ((self.head_dim // 3) // 2)
        )
        self.grid_size = grid_size

    def _separate_positions(self, ids, H, W):
        tokens_per_frame = H * W
        frame_ids = ids // tokens_per_frame
        rem = ids - tokens_per_frame * frame_ids
        height_ids = rem // W
        width_ids = rem - W * height_ids
        return frame_ids.float(), height_ids.float(), width_ids.float()

    def forward(
        self,
        x,
        mask=None,
        attn_mask=None,
        D_patches=None,
        H_patches=None,
        W_patches=None,
    ):
        B, N, C = x.size()
        qkv = (
            self.qkv(x)
            .unflatten(-1, (3, self.num_heads, self.head_dim))
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if mask is not None:
            pos = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
        else:
            pos = torch.arange(
                D_patches * H_patches * W_patches, device=x.device
            )
        d_pos, h_pos, w_pos = self._separate_positions(
            pos, H_patches, W_patches
        )

        s = 0
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], d_pos)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], d_pos)
        s += self.d_dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], h_pos)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], h_pos)
        s += self.h_dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], w_pos)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], w_pos)
        s += self.w_dim

        if s < self.head_dim:
            q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
            k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop_p if self.training else 0.0
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if math.isclose(self.drop_prob, 0.0) or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask


class Block(nn.Module):
    """One Transformer block: (RoPE-)Attention + residual, then either a
    dense MLP or a sparse MoE + residual."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        use_moe=False,
        moe_params: Optional[MoEParams] = None,
        grid_size=16,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RoPEAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            grid_size=grid_size,
        )
        self.drop_path = (
            nn.Identity()
            if math.isclose(drop_path, 0.0)
            else DropPath(drop_path)
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        if use_moe:
            assert moe_params is not None
            self.mlp = MoE(
                dim=dim,
                n_shared_experts=moe_params.n_shared_experts,
                n_routed_experts=moe_params.n_routed_experts,
                n_activated_experts=moe_params.n_activated_experts,
                moe_inter_dim=moe_params.moe_inter_dim,
                score_func=moe_params.score_func,
                route_scale=moe_params.route_scale,
                drop=drop,
            )
        else:
            self.mlp = MLP(dim, mlp_hidden, drop=drop)

    def forward(
        self, x, mask=None, D_patches=None, H_patches=None, W_patches=None
    ):
        y = self.attn(
            self.norm1(x),
            mask=mask,
            D_patches=D_patches,
            H_patches=H_patches,
            W_patches=W_patches,
        )
        x = x + self.drop_path(y)
        moe_scores = None
        if isinstance(self.mlp, MoE):
            y, moe_scores = self.mlp(self.norm2(x))
        else:
            y = self.mlp(self.norm2(x))
        x = x + self.drop_path(y)
        return x, moe_scores


class VisionTransformer3DMoE(nn.Module):
    """3D ViT backbone with an optional sparse Mixture-of-Experts (MoE)
    mixed in at configurable layers.

    Parameters
    ----------
    img_size : (int, int, int), default=(96, 108, 96)
        Size (in voxels) of the input volume.
    patch_size : (int, int, int), default=(12, 12, 12)
        Size (in voxels) of one cubic patch ("tubelet").
    in_chans : int, default=1
        Number of input channels.
    embed_dim : int, default=768
        Token embedding dimension.
    depth : int, default=12
        Number of Transformer blocks.
    num_heads : int, default=12
        Number of attention heads.
    mlp_ratio : float, default=4.0
        Ratio between the MLP hidden dimension and `embed_dim` (dense
        blocks only; MoE blocks use `moe_params.moe_inter_dim` instead).
    drop_path_rate : float, default=0.0
        Maximum stochastic-depth drop rate, linearly increased across
        blocks.
    use_moe : bool, default=False
        Whether to replace the dense MLP with a sparse MoE (see `MoE`) in
        the blocks listed in `moe_params.moe_layer_indices`.
    moe_params : MoEParams or None, default=None
        Sparse MoE hyperparameters. Required if `use_moe=True`.
    init_std : float, default=0.02
        Standard deviation used for truncated-normal weight init.
    """

    def __init__(
        self,
        img_size=(96, 108, 96),
        patch_size=(12, 12, 12),
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_path_rate=0.0,
        use_moe=False,
        moe_params: Optional[MoEParams] = None,
        init_std=0.02,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.img_size = img_size
        self.use_moe = use_moe
        self.moe_layer_indices = (
            list(moe_params.moe_layer_indices)
            if (use_moe and moe_params)
            else []
        )

        self.patch_embed = PatchEmbed3D(patch_size, in_chans, embed_dim)
        self._grid_shape = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[i],
                    use_moe=(use_moe and i in self.moe_layer_indices),
                    moe_params=moe_params,
                    grid_size=self._grid_shape[0],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        """(nH, nW, nD) patch-grid dimensions."""
        return self._grid_shape

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        # Deeper layers get progressively smaller init (GPT-2 trick), applied
        # per-expert when MoE is on -- mirrors the official model.
        for layer_id, blk in enumerate(self.blocks):
            blk.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            if isinstance(blk.mlp, MoE):
                for expert in blk.mlp.experts:
                    expert.fc2.weight.data.div_(
                        math.sqrt(2.0 * (layer_id + 1))
                    )
                blk.mlp.shared_experts.fc2.weight.data.div_(
                    math.sqrt(2.0 * (layer_id + 1))
                )
            else:
                blk.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def forward(
        self, x: torch.Tensor, masks: Optional[list[torch.Tensor]] = None
    ):
        """Encode a volume into patch tokens.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, C, H, W, D)`` volume.
        masks : list of torch.Tensor or None, default=None
            Optional list of ``(B, K)`` LongTensors: if given, only those
            patch indices are kept, and the output batch is multiplied by
            ``len(masks)`` (one block per mask, concatenated along batch).

        Returns
        -------
        tokens : torch.Tensor
            ``(B[*len(masks)], K, E)`` patch tokens.
        moe_scores : list
            One entry per block of router scores, or an empty list if
            `use_moe=False`.
        """
        _, _, H, W, D = x.shape
        H_p, W_p, D_p = (
            H // self.patch_size[0],
            W // self.patch_size[1],
            D // self.patch_size[2],
        )

        x = self.patch_embed(x)  # (B, N, E)

        pos_for_attn = None
        if masks is not None:
            x = apply_masks(x, masks)  # (B*len(masks), K, E)
            pos_for_attn = torch.cat(
                masks, dim=0
            )  # (B*len(masks), K) absolute patch indices

        moe_scores_all = []
        for blk in self.blocks:
            x, moe_scores = blk(
                x,
                mask=pos_for_attn,
                D_patches=D_p,
                H_patches=H_p,
                W_patches=W_p,
            )
            if self.use_moe:
                moe_scores_all.append(moe_scores)

        return self.norm(x), moe_scores_all
