##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

import math
from multiprocessing import Value
from typing import Any, Optional, Union

import torch
from einops import rearrange
from pytorch_lightning.utilities.types import LRSchedulerPLType
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block, VisionTransformer
from torch import nn
from torch.optim import Optimizer

from nidl.estimators.base import BaseEstimator, TransformerMixin
from nidl.volume.backbones.tokenizers.patchify import (
    build_2d_sincos_posemb,
    build_3d_sincos_posemb,
)

from .utils.data_parsing import parse_x_or_xy_batch
from .utils.encoder import build_encoder
from .utils.momentum import MomentumUpdater, initialize_momentum_params


class IJEPA(TransformerMixin, BaseEstimator):
    """Implementation of I-JEPA [1]_.

    Solver that predicts the representations of missing parts of an image based
    on its surrounding context. It uses two encoders (context encoder and
    target encoder) to obtain the contextual and target features and a third
    predictor network to obtain the predictions.

    The target encoder is an Exponential Moving Average (EMA) of the context
    encoder. It is used at inference time, the context encoder and predictor
    being dropped.


    Parameters
    ----------
    encoder : timm.models.vision_transformer.VisionTransformer
        The encoder architecture.
    
    dim: {2, 3}, default=3
        Input data dimensionality. 3 == 3d volumes and 2 == 2d images.

    context_block_scale : (float, float), default=(0.85, 1.0)
        Range of scale of the context block.
    
    target_block_scale : (float, float), default=(0.15, 0.2)
        Range of scale of the target blocks.

    aspect_ratio : (float, float), default=(0.75, 1.5)
        Aspect ratio of the target blocks.
        
    num_target_blocks : int, default=4
        Number of target blocks to predict.

    min_keep : int, default=4
        Minimum number of patches to keep in the context/target block.

    allow_overlap : bool, default=False
        Whether to allow overlap between target and context blocks.

    predictor_embed_dim : int, default=384
        Dimension of the predictor hidden layers. It can be different from
        the encoder output dimension.

    predictor_depth_pred : int, default=6
        Number of Transformer blocks in the predictor.

    optimizer : {'sgd', 'adam', 'adamW'} or Optimizer, default="adamW"
        Optimizer for training the model. If a string is given, it can be:

            - 'sgd': Stochastic Gradient Descent (with optional momentum).
            - 'adam': First-order gradient-based optimizer.
            - 'adamW' (default): Adam with decoupled weight decay
              regularization (see "Decoupled Weight Decay Regularization",
              Loshchilov and Hutter, ICLR 2019).

    learning_rate : float, default=3e-4
        Initial learning rate.

    weight_decay : float, default=5e-4
        Weight decay in the optimizer.

    exclude_bias_and_norm_wd : bool, default=True
        Whether the bias terms and normalization layers get weight decay during
        optimization or not.

    ema_start : float, default=0.996
        Base value for the weighting coefficient in the teacher momentum
        update with exponential moving average. A cosine annealing scheme is
        used.

    ema_end : float, default=1.0
        Final value for the weighting coefficient in the teacher momentum
        update.

    optimizer_kwargs : dict or None, default=None
        Extra named arguments for the optimizer.

    lr_scheduler : {"none", "warmup_cosine"}, LRSchedulerPLType or None,\
        default="warmup_cosine"
        Learning rate scheduler to use.

    lr_scheduler_kwargs : dict or None, default=None
        Extra named arguments for the scheduler. By default, it is set to
        {"warmup_epochs": 10, "warmup_start_lr": 1e-6, "min_lr": 0.0,
        "interval": "step"}

    **kwargs : dict, optional
        Extra named arguments for the BaseEstimator class (given to
        PL Trainer), such as `max_epochs`, `max_steps`, `num_sanity_val_steps`,
        `check_val_every_n_epoch`, `callbacks`, etc.
        See the PL `Trainer API <https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api>`_
        for more details.

    Attributes
    ----------
    encoder : torch.nn.Module
        It corresponds to the target encoder in the I-JEPA model.

    predictor : torch.nn.Module
        Predictor model trained to predict the masked part of an image from a
        context in the latent space. This can be useful at inference time to
        predict missing part of the input.

    loss : SmoothL1Loss
        Smooth l1 loss used for training the model.

    References
    ----------
    .. [1] Self-Supervised Learning from Images with a Joint-Embedding
           Predictive Architecture, Assran et al., ICCV 2023
    """

    def __init__(
        self,
        encoder: VisionTransformer,
        dim: int = 3,
        context_block_scale: tuple[float, float] = (0.85, 1.0),
        target_block_scale: tuple[float, float] = (0.15, 0.2),
        aspect_ratio: tuple[float, float] = (0.75, 1.5),
        num_target_blocks: int = 4,
        min_keep: int = 4,
        allow_overlap: bool = False,
        predictor_embed_dim: int = 384,
        predictor_depth_pred: int = 6,
        optimizer: Union[str, Optimizer] = "adamW",
        learning_rate: float = 3e-4,
        weight_decay: float = 5e-4,
        exclude_bias_and_norm_wd: bool = True,
        ema_start: float = 0.996,
        ema_end: float = 1.0,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        lr_scheduler: Optional[
            Union[str, LRSchedulerPLType]
        ] = "warmup_cosine",
        lr_scheduler_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):

        ignore = kwargs.pop("ignore", ["callbacks", "encoder"])
        if "callbacks" not in ignore:
            ignore.append("callbacks")
        if "encoder" not in ignore:
            ignore.append("encoder")

        super().__init__(**kwargs)

        self.parse_batch = parse_x_or_xy_batch

        self.target_encoder = IJEPAVisionTransformer(build_encoder(encoder))
        self.context_encoder = IJEPAVisionTransformer(
            build_encoder(encoder, deepcopy=True)
        )

        # Copy context encoder params to target encoder + no_grad for target.
        initialize_momentum_params(self.context_encoder, self.target_encoder)

        self.encoder = self.target_encoder.vit

        # Instantiate momentum updates.
        self.momentum_updater = MomentumUpdater(ema_start, ema_end)

        # Derive input parameters from given encoder
        grid_size = self.encoder.patch_embed.grid_size
        embed_dim = self.encoder.embed_dim
        num_heads = self.encoder.blocks[0].attn.num_heads

        self.predictor = VisionTransformerPredictor(
            grid_size=grid_size,
            dim=dim,
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth_pred,
            num_heads=num_heads,
        )

        self.masker = Masker(
            grid_size=grid_size,
            dim=dim,
            context_block_scale=context_block_scale,
            target_block_scale=target_block_scale,
            aspect_ratio=aspect_ratio,
            num_target_blocks=num_target_blocks,
            min_keep=min_keep,
            allow_overlap=allow_overlap,
        )

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.exclude_bias_and_norm_wd = exclude_bias_and_norm_wd
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = (
            lr_scheduler_kwargs if lr_scheduler_kwargs is not None else {}
        )
        self.optimizer_kwargs = optimizer_kwargs

        self.loss = nn.SmoothL1Loss()

        self._fill_default_lr_scheduler_kwargs()

    def _shared_step(self, batch):
        x, _ = self.parse_batch(batch, device=self.device)

        context_block, target_blocks = self.masker(len(x))

        # Compute target features
        h = self.forward_target(x, target_blocks)

        # Compute context features
        z = self.context_encoder(x, [context_block])

        # Predict target features from context features
        z = self.predictor(z, context_block, target_blocks)

        # Compute predictive loss
        loss = self.loss(z, h)

        return {
            "loss": loss,
            "z_context": z.detach(),
            "z_target": h.detach(),
        }

    def training_step(self, batch, batch_idx: int):
        """Perform one training step during an epoch and computes the
        training loss.

        Parameters
        ----------
        batch : torch.Tensor or pair of torch.Tensor
            A batch of data in the format ``X`` or ``(X, Y)`` where ``X``
            is a torch.Tensor with shape ``(B, C, H, W)`` (2d images) or
            ``(B, C, H, W, D)`` (3d volumes) representing the input data.
            `Y` are eventual labels (ignored).

        batch_idx : int
            The index of the current batch (ignored).

        Returns
        -------
        outputs : dict
            Dictionary containing three torch.Tensors:
                - "loss": training loss computed on this batch of data.
                - "z_context": embeddings predictions for target tokens with
                   shape ``(B * M, L, D)`` where ``M`` is the number of
                   target blocks (e.g. 4), ``L`` is the number of target
                   tokens predicted per block, and ``D`` is the embedding
                   dimension.
                - "z_target": embeddings to predict with the same shape as
                  "z_context".
        """

        outputs = self._shared_step(batch)
        self.log("loss/train", outputs["loss"], prog_bar=True, sync_dist=True)
        # Returns everything needed for further logging/metrics computation
        return outputs

    @torch.no_grad()
    def forward_target(
        self,
        x: torch.Tensor,
        target_blocks: list[torch.Tensor],
    ):
        h = self.target_encoder.forward_features(x)
        # create targets (only preserve tokens in the target blocks).
        h = IJEPAVisionTransformer.apply_masks(h, target_blocks)
        return h

    def on_train_batch_end(
        self, outputs: dict[str, Any], batch, batch_idx: int
    ):
        """Performs the teacher momentum update.

        Parameters
        ----------
        outputs : dict[str, Any]
            The outputs of the training step (ignored).
        batch : torch.Tensor or pair of torch.Tensor
            A batch of input data (ignored).
        batch_idx : int
            The index of the current batch (ignored).
        """

        # update target backbone
        self.momentum_updater.update(self.context_encoder, self.target_encoder)
        # log lambda momentum
        self.log("lambda", self.momentum_updater.cur_lambda)
        # update lambda
        self.momentum_updater.update_lambda(
            cur_step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
        )

    def validation_step(self, batch: Any, batch_idx: int):
        """Performs one validation step and computes the validation
        loss.

        Parameters
        ----------
        batch : torch.Tensor or pair of torch.Tensor
            A batch of data in the format ``X`` or ``(X, Y)`` where ``X``
            is a torch.Tensor with shape ``(B, C, H, W)`` (2d images) or
            ``(B, C, H, W, D)`` (3d volumes) representing the input data.
            `Y` are eventual labels (ignored).

        batch_idx : int
            The index of the current batch (ignored).

        Returns
        -------
        outputs : dict
            Dictionary containing three torch.Tensors:
                - "loss": validation loss computed on this batch of data.
                - "z_context": embeddings predictions for target tokens with
                   shape ``(B * M, L, D)`` where ``M`` is the number of
                   target blocks (e.g. 4), ``L`` is the number of target
                   tokens predicted per block, and ``D`` is the embedding
                   dimension.
                - "z_target": embeddings to predict with the same shape as
                  "z_context".
        """

        outputs = self._shared_step(batch)
        self.log("loss/val", outputs["loss"], prog_bar=True, sync_dist=True)
        # Returns everything needed for further logging/metrics computation
        return outputs

    def _fill_default_lr_scheduler_kwargs(self):
        if self.lr_scheduler_kwargs is None:
            self.lr_scheduler_kwargs = {}

        self.lr_scheduler_kwargs.setdefault("warmup_epochs", 10)
        self.lr_scheduler_kwargs.setdefault("interval", "step")
        self.lr_scheduler_kwargs.setdefault("warmup_start_lr", 1e-6)
        self.lr_scheduler_kwargs.setdefault("min_lr", 0)


class IJEPAVisionTransformer(nn.Module):
    def __init__(self, vit: VisionTransformer):
        super().__init__()
        if not isinstance(vit, VisionTransformer):
            raise TypeError(
                "Encoder must be a VisionTransformer from timm in I-JEPA, "
                f"got {vit}."
            )

        self.vit = vit
        # Raise error if class/reg tokens are present (not used in I-JEPA)
        if self.vit.has_class_token or self.vit.num_reg_tokens > 0:
            raise ValueError(
                "VisionTransformer in I-JEPA should not have CLS or REG tokens"
            )

    def forward(
        self, x: torch.Tensor, masks: Optional[list[torch.Tensor]] = None
    ):
        """Forward pass through feature layers (embeddings, transformer blocks,
        post-transformer norm) + masking of input patches."""
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)

        if masks is not None:
            x = self.apply_masks(x, masks)

        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x

    def forward_features(self, x: torch.Tensor):
        return self.vit.forward_features(x)

    @staticmethod
    def apply_masks(x: torch.Tensor, masks: list[torch.Tensor]):
        """
        Applies patch-wise masks to a batched input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, N, D)``, where:
                - ``B`` is the batch size,
                - ``N`` is the number of input patches,
                - ``D`` is the embedding dimension.

        masks : List[torch.Tensor]
            List of ``M`` tensors, each of shape ``(B, K)``, where:
                - ``M`` is the number of blocks,
                - ``B`` is the batch size,
                - ``K`` is the number of patches to keep.
            Each tensor contains the indices of the patches to retain in each
            sample of the batch.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B * M, K, D),`` representing the retained
            patches in the batch.
        """
        all_x = []
        for m in masks:
            mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
            all_x += [torch.gather(x, dim=1, index=mask_keep)]
        return torch.cat(all_x, dim=0)


class VisionTransformerPredictor(nn.Module):
    """Lightweight Vision Transformer taking as input a sequence of tokens."""

    def __init__(
        self,
        grid_size: Union[int, tuple[int, int], tuple[int, int, int]],
        dim: int = 3,
        embed_dim=192,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
    ):
        super().__init__()
        self.dim = dim
        self.grid_size = _parse_grid_size(grid_size, dim)

        h, w = self.grid_size[:2]
        if dim == 3:
            d = self.grid_size[2]

        self.predictor_embed = nn.Linear(
            embed_dim, predictor_embed_dim, bias=True
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        num_patches = math.prod(self.grid_size)
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim),
            requires_grad=False,
        )

        if dim == 2:
            predictor_pos_embed = rearrange(
                build_2d_sincos_posemb(
                    h=h, w=w, embed_dim=predictor_embed_dim
                ),
                "b d nh nw -> b (nh nw) d",
            )
        elif dim == 3:
            predictor_pos_embed = rearrange(
                build_3d_sincos_posemb(
                    h=h, w=w, d=d, embed_dim=predictor_embed_dim
                ),
                "b d nh nw nd -> b (nh nw nd) d",
            )

        self.predictor_pos_embed.data.copy_(predictor_pos_embed)

        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(
            predictor_embed_dim, embed_dim, bias=True
        )
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        context_block: torch.Tensor,
        target_blocks: list[torch.Tensor],
    ):
        """
        Perform forward pass of the I-JEPA predictor to infer target token
        representations from contextual input tokens.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, K, D)``, where ``B`` is the batch
            size,``K`` is the number of context tokens, and ``D`` is the
            embedding dimension.

        context_block : torch.Tensor
            Tensor containing indices of the input (context) tokens given to
            the context encoder.

        target_blocks : List[torch.Tensor]
            List of ``M`` tensors containing indices of the target tokens to
            predict from the context tokens. ``M`` is the number of target
            blocks to predict from a single context.

        Returns
        -------
        x : torch.Tensor
            Output tensor of shape ``(B * M, L, D)``, where ``B`` is
            the batch size, ``M`` is the number of target blocks, ``L``
            is the number of target tokens in each block, and ``D`` is the
            embedding dimension
        """

        B = len(x)
        M = len(target_blocks)

        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)  # shape [B, K, D']

        # -- add positional embedding to contextual tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += IJEPAVisionTransformer.apply_masks(
            x_pos_embed, [context_block]
        )  # shape [B, K, D']

        _, K, _ = x.shape

        # -- create mask tokens and append it to contextual tokens
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = IJEPAVisionTransformer.apply_masks(
            pos_embs, target_blocks
        )  # shape [B * M, L, D']

        BM, L, _ = pos_embs.shape

        pred_tokens = self.mask_token.repeat(BM, L, 1)
        # -- add positional embeddings to mask tokens
        pred_tokens += pos_embs
        x = x.repeat(M, 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = x[:, K:]
        x = self.predictor_proj(x)

        return x


class Masker(nn.Module):
    """Masker of the JEPA model yielding indices of the context/target blocks.

    In details, for each forward pass, it yields:

    1) One context block per sample given to a context encoder with random
        scale and unit aspect ratio.
    2) ``num_target_blocks`` (e.g. 4) target blocks per sample given to a
       target encoder with random scale and random aspect ratio.

    It handles either 2D images or 3D volumes as input and generates flattened
    2D (respectively 3D) indices of the context/target blocks as 1D vectors.

    This implementation is adapted from the original I-JEPA implementation in :
    https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py

    Parameters
    ----------
    grid_size: int, (int, int) or (int, int, int)
        Grid size over the inputs (height, width) or (height, width, depth).

    dim: int, default=3
        Dimensionality of the input, 2 == "2d" image and 3 == "3d" volume.

    context_block_scale: (float, float), default=(0.85, 1.0)
        Range of the scale of the context block (unit aspect ratio).

    target_block_scale: (float, float), default=(0.15, 0.2)
        Range of the scale of the target blocks.

    aspect_ratio: (float, float), default=(0.75, 1.5)
        Range of the aspect ratio of target blocks.

    num_target_blocks: int, default=4
        Number of target blocks to yield per sample.

    min_keep: int, default=4
        Minimum number of patches to keep in the context/target block.

    allow_overlap: bool, default=False
        Whether to allow overlap between target and context blocks.


    Examples
    --------
    >>> import torch
    >>> # 2D: 14x14 patch grid (e.g. 224x224 with 16x16 patches)
    >>> masker2d = Masker(grid_size=(14, 14), dim=2, num_target_blocks=4,
    ... min_keep=4)
    >>> context, targets = masker2d(batch_size=2)
    >>> context.shape
    torch.Size([2, Kc])
    >>> targets.shape
    torch.Size([4, 2, Kt])  # (num_target_blocks, batch, tokens_per_block)
    >>> context.dtype, targets.dtype
    (torch.int64, torch.int64)

    >>> # 3D: 8x8x8 patch grid (e.g. 128x128x128 with 16x16x16 patches)
    >>> masker3d = Masker(grid_size=(8, 8, 8), dim=3, num_target_blocks=2,
    ... min_keep=8)
    >>> context3, targets3 = masker3d(batch_size=1)
    >>> context3.shape, targets3.shape
    (torch.Size([1, Kc]), torch.Size([2, 1, Kt]))

    Notes
    -----
    An input 2D image or 3D volume is seen as an ordered sequence of small
    non-overlapping patches and the blocks are defined over this grid of
    patches (following the ViT paradigm). As such, returned indices are defined
    over this grid and NOT over input pixels/voxels.

    """

    def __init__(
        self,
        grid_size: Union[int, tuple[int, int], tuple[int, int, int]],
        dim: int = 3,
        context_block_scale: tuple[float, float] = (0.85, 1.0),
        target_block_scale: tuple[float, float] = (0.15, 0.2),
        aspect_ratio: tuple[float, float] = (0.75, 1.5),
        num_target_blocks: int = 4,
        min_keep: int = 4,
        allow_overlap: bool = False,
    ):
        super().__init__()

        self.grid_size = _parse_grid_size(grid_size, dim)
        self.dim = dim
        self.context_block_scale = context_block_scale
        self.target_block_scale = target_block_scale
        self.aspect_ratio = aspect_ratio
        self.num_context_block = 1  # Fixed
        self.num_target_blocks = num_target_blocks
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self._itr_counter = Value("i", -1)

    def forward(self, batch_size):
        """Yields masks for one context block and multiple target blocks.

        Strategy for creating the blocks:
            1. sample context block size using seed
            2. sample target block size using seed
            3. sample one context block location for each image (w/o seed)
            4. sample several target block locations for each image (w/o seed)
            5. return context mask and target mask

        Parameters
        ----------
        batch_size: int
            Number of samples in the current batch.

        Returns
        ----------
        (context_block, target_blocks): torch.Tensor, List[torch.Tensor]
            ``context_block`` has shape ``(B, Kc)`` where ``B`` is the batch
              size and ``Kc`` is the number of context patch indices kept.
            ``target_blocks`` has shape ``(M, B, Kt)`` where ``M`` is the
              number of target blocks and ``Kt`` is the number of target patch
              indices kept per block.

        Notes
        -----
        The number of kept tokens per block can vary across samples because
        blocks are randomly placed and constrained by `min_keep` and
        `allow_overlap`. To return tensors with a consistent shape, the masker
        truncates each sampled mask to the minimum number of kept indices
        observed in the batch:

        - `Kc` is the minimum number of context indices across the batch.
        - `Kt` is the minimum number of target indices across the batch (and
          across all target blocks).

        As a result, `context` has shape `(B, Kc)` and `targets` has shape
        `(M, B, Kt)`, where `B` is the batch size and `M` is
        `num_target_blocks`.

        """
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.target_block_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )
        e_size = self._sample_block_size(
            generator=g,
            scale=self.context_block_scale,
            aspect_ratio_scale=(1.0, 1.0),
        )

        collated_masks_target, collated_masks_context = [], []
        min_keep_target = math.prod(self.grid_size)
        min_keep_context = math.prod(self.grid_size)

        for _ in range(batch_size):
            masks_p, masks_C = [], []
            for _ in range(self.num_target_blocks):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_target = min(min_keep_target, len(mask))
            collated_masks_target.append(masks_p)

            acceptable_regions = masks_C

            if self.allow_overlap:
                acceptable_regions = None

            mask, _ = self._sample_block_mask(
                e_size, acceptable_regions=acceptable_regions
            )
            min_keep_context = min(min_keep_context, len(mask))
            collated_masks_context.append(mask)

        # Masks with shape ``num_target_blocks x batch_size x min_keep_target``
        collated_masks_target = [
            [cm[:min_keep_target] for cm in cm_list]
            for cm_list in collated_masks_target
        ]
        collated_masks_target = torch.utils.data.default_collate(
            collated_masks_target
        )

        # Mask with shape ``batch_size x min_keep_context``
        collated_masks_context = [
            cm[:min_keep_context] for cm in collated_masks_context
        ]
        collated_masks_context = torch.utils.data.default_collate(
            collated_masks_context
        )

        return (
            collated_masks_context,
            collated_masks_target,
        )

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        # Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(math.prod(self.grid_size) * mask_scale)
        # Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # Compute block size (given scale and aspect-ratio)
        axis1 = round(math.pow(max_keep * aspect_ratio, 1.0 / self.dim))
        axis2 = round(math.pow(max_keep / aspect_ratio, 1.0 / self.dim))
        block_size = [axis1, axis2]
        if self.dim == 3:
            axis3 = round(math.pow(max_keep, 1 / self.dim))
            block_size.append(axis3)
            perm = torch.randperm(3, generator=generator)
            block_size = block_size[perm]
        # Constrain block size to be smaller than the input size
        for i, gs in enumerate(self.grid_size):
            block_size[i] = min(block_size[i], gs - 1)
        return tuple(block_size)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        # Generates a mask at a random location as a tensor of indices
        def constrain_mask(mask, tries=0):
            """Helper to restrict given mask to a set of acceptable regions"""
            N = max(int(len(acceptable_regions) - tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]

        # Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # Sample start position for each dimension
            starts = []
            for grid, block in zip(self.grid_size, b_size):
                start = torch.randint(0, grid - block + 1, (1,)).item()
                starts.append(start)
            # Create the mask
            mask = torch.zeros(self.grid_size, dtype=torch.int32)
            slices = tuple(slice(s, s + b) for s, b in zip(starts, b_size))
            mask[slices] = 1
            # Constrain the mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask_complement = ~mask.bool().clone().to(torch.int32)
            mask = torch.nonzero(mask.flatten())
            # If mask is too small, try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    print(
                        'Mask generator says: "Valid mask not found,'
                        f"decreasing acceptable-regions [{tries}]"
                    )
        mask = mask.squeeze()
        return mask, mask_complement


def _parse_grid_size(grid_size, dim):
    if dim == 2:
        if isinstance(grid_size, (list, tuple)) and len(grid_size) == 2:
            return tuple(grid_size)
        elif isinstance(grid_size, int):
            return (grid_size, grid_size)
        raise ValueError(
            "grid_size must be a pair of (int, int) or a single int for "
            "2d images."
        )
    elif dim == 3:
        if isinstance(grid_size, (list, tuple)) and len(grid_size) == 3:
            return tuple(grid_size)
        elif isinstance(grid_size, int):
            return (grid_size, grid_size, grid_size)
        raise ValueError(
            "grid_size must be a triplet of (int, int, int) or a single "
            "int for 3d volumes."
        )
    raise ValueError(f"dim must be 2 or 3, got {dim}")
