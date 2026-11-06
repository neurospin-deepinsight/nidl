##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Optional, Union

import torch
from einops import repeat
from pytorch_lightning.utilities.types import LRSchedulerPLType
from torch import nn
from torch.optim import Optimizer

from nidl.backbones.volume.utils import SimpleFeaturesInputAdapter
from nidl.estimators.base import BaseEstimator, TransformerMixin
from nidl.estimators.ssl.utils.encoder import build_encoder
from nidl.estimators.ssl.utils.optimizer import configure_ssl_optimizers
from nidl.estimators.ssl.utils.projection_heads import CoMMProjectionHead
from nidl.losses import CoMMLoss
from nidl.utils.data_parsing import gather_tensor, parse_multimodal_batch


class QuickGELU(nn.Module):
    """Fast approximation of the GELU activation, as in CLIP.

    Adapted from https://github.com/Duplums/CoMM.
    """

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Self-attention block.

    Adapted from https://github.com/Duplums/CoMM.
    """
    def __init__(self, d_model: int, num_heads: int,
                 add_bias_kv: bool = False,
                 attn_drop_rate: float = 0.,
                 batch_first: bool = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads,
                                          add_bias_kv=add_bias_kv,
                                          dropout=attn_drop_rate,
                                          batch_first=batch_first)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor,
                  key_padding_mask: Optional[torch.Tensor] = None):
        return self.attn(x.clone(), x, x, need_weights=False,
                         key_padding_mask=key_padding_mask)[0]

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.ln_1(x),
                               key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class FusionTransformer(nn.Module):
    """Fusion of features from multiple modalities using attention:
    concatenation over tokens + self-attention module.

    in_shape: (N, L1, E), (N, L2, E), out_shape: (N, E). An attention mask can
    be applied for each modality with shape (N, Li) for modality i.

    Adapted from https://github.com/Duplums/CoMM.

    Parameters
    ----------
    width : int
        Embedding size.
    num_heads : int
        Number of heads in multi-head attention blocks.
    depth : int
        Number of attention blocks.
    add_bias_kv : bool, default=False
        If True, adds bias to the key and value sequences at dim=0.
    attn_drop_rate : float, default=0.0
        Dropout probability on `attn_output_weights`.
    batch_first : bool, default=True
        Whether the input tensors are (batch, tokens, features) (True) or
        (tokens, batch, features) (False).
    """
    def __init__(self, width: int,
                 num_heads: int,
                 depth: int,
                 add_bias_kv: bool = False,
                 attn_drop_rate: float = 0.,
                 batch_first: bool = True):
        super().__init__()

        self.width = width
        self.layers = depth
        self.norm = nn.LayerNorm(width)
        self.token_dim = 1 if batch_first else 0
        self.cls_token = nn.Parameter(torch.randn(1, 1, width))
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, num_heads, add_bias_kv=add_bias_kv,
                                   attn_drop_rate=attn_drop_rate,
                                   batch_first=batch_first)
            for _ in range(depth)])
        self.initialize()

    def initialize(self):
        proj_std = (self.width ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        for block in self.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(self, x: list[torch.Tensor],
                key_padding_mask: Optional[list[torch.Tensor]] = None):
        """Fuse the input tensors (N, Li, E) into (N, E), with optional bool
        masks (N, Li), one per input tensor, where `True` indicates
        unattended tokens.
        """
        # Concatenate over tokens + self-attention
        x = torch.cat(x, dim=self.token_dim)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(key_padding_mask, dim=self.token_dim)
        # append cls token at the beginning
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
        x = torch.cat((cls_token, x), dim=self.token_dim)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                (torch.zeros_like(cls_token[:, :, 0]), key_padding_mask),
                dim=self.token_dim)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.masked_fill(
                key_padding_mask.bool(), float("-inf")).float()

        for layer in self.resblocks:
            x = layer(x, key_padding_mask=key_padding_mask)

        x = self.norm(x)

        x = x[:, 0] if self.token_dim == 1 else x[0]
        return x


class MissingModMMFusion(nn.Module):
    """MM fusion over a batch, with per-sample missing modalities.

    Every present-modality volume is concatenated, plus `sample_idx` /
    `mod_idx` recording which (sample, modality) each entry belongs to. Each
    backbone is run over its own modality slots in a single forward pass,
    tokens are scattered back into a (B, M, T, D) grid, and absent (sample,
    modality) cells are masked out of the `FusionTransformer`.

    Adapted from the MMFusion of https://github.com/Duplums/CoMM.

    Parameters
    ----------
    encoder : nn.Module or sequence of nn.Module
        Backbone, or list of backbones (e.g. one for structural modalities
        and one for diffusion). With several backbones, `mod_to_encoder` says
        which one encodes each modality slot. All backbones must output
        feature maps of the same shape, since a single `adapter` tokenizes
        them.
    adapter : nn.Module
        Tokenizer for the encoder output.
    num_modalities : int
        Number of modality slots.
    mod_to_encoder : sequence of int or None, default=None
        (M,) index of the encoder handling each modality slot. Defaults to
        encoder 0 for every slot (single shared backbone).
    embed_dim : int, default=2048
        Fusion width; must equal the adapter's token dim.
    freeze_encoder : bool, default=True
        Whether to freeze or train the backbone encoders.
    per_modality_adapter : bool, default=False
        Give each modality slot its own copy of `adapter` instead of sharing
        one.
    num_heads, depth, add_bias_kv, attn_drop_rate
        `FusionTransformer` hyper-parameters.
    """
    def __init__(self,
                 encoder: Union[nn.Module, Sequence[nn.Module]],
                 adapter: nn.Module,
                 num_modalities: int,
                 mod_to_encoder: Optional[Sequence[int]] = None,
                 embed_dim: int = 2048,
                 freeze_encoder: bool = True,
                 per_modality_adapter: bool = False,
                 num_heads: int = 8,
                 depth: int = 1,
                 add_bias_kv: bool = False,
                 attn_drop_rate: float = 0.):
        super().__init__()
        encoders = ([encoder] if isinstance(encoder, nn.Module)
                    else list(encoder))
        self.encoders = nn.ModuleList(encoders)
        self.adapters = self._build_adapters(
            adapter, num_modalities if per_modality_adapter else 1)
        self.num_modalities = num_modalities
        self.register_buffer(
            "mod_to_encoder",
            self._check_mod_to_encoder(mod_to_encoder, num_modalities,
                                       len(self.encoders)))
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoders.parameters():
                p.requires_grad = False
            self.encoders.eval()
        self.fusion_transformer = FusionTransformer(
            embed_dim, num_heads, depth, add_bias_kv, attn_drop_rate,
            batch_first=True)

    @staticmethod
    def _build_adapters(adapter: nn.Module, num_copies: int) -> nn.ModuleList:
        """`num_copies` tokenizers: a single common adapter for all
        modalities when 1, clones when more.
        """
        if num_copies == 1:
            return nn.ModuleList([adapter])
        adapters = nn.ModuleList()
        for _ in range(num_copies):
            clone = copy.deepcopy(adapter)
            for module in clone.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            adapters.append(clone)
        return adapters

    @staticmethod
    def _check_mod_to_encoder(mod_to_encoder: Optional[Sequence[int]],
                              num_modalities: int,
                              num_encoders: int) -> torch.Tensor:
        """Modality slot -> encoder index, as a (M,) long tensor."""
        if mod_to_encoder is None:
            if num_encoders > 1:
                raise ValueError(
                    "`mod_to_encoder` is required with several encoders "
                    f"(got {num_encoders}).")
            return torch.zeros(num_modalities, dtype=torch.long)
        mod_to_encoder = torch.as_tensor(list(mod_to_encoder),
                                         dtype=torch.long)
        if mod_to_encoder.numel() != num_modalities:
            raise ValueError(
                f"`mod_to_encoder` has {mod_to_encoder.numel()} entries for "
                f"{num_modalities} modalities.")
        if (int(mod_to_encoder.max()) >= num_encoders
                or int(mod_to_encoder.min()) < 0):
            raise ValueError(
                f"`mod_to_encoder`={mod_to_encoder.tolist()} is out of range "
                f"for {num_encoders} encoder(s).")
        return mod_to_encoder

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:  # keep frozen backbones (incl. BN) in eval
            self.encoders.eval()
        return self

    @staticmethod
    def _apply_by_group(modules: nn.ModuleList, x: torch.Tensor,
                        group_idx: torch.Tensor) -> torch.Tensor:
        """Run `modules[g]` over the rows of `x` (N, ...) with
        `group_idx == g`, then restore the original row order.
        """
        outs, rows = [], []
        for g, module in enumerate(modules):
            # (N_g,) rows in group g
            sel = (group_idx == g).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:          # group absent from this batch
                continue
            outs.append(module(x.index_select(0, sel)))       # (N_g, ...)
            rows.append(sel)                                  # (N_g,)
        # scatter the per-group slices back into the original row order;
        # sum(N_g) == N, so every row of `out` is written exactly once
        out = outs[0].new_zeros(x.shape[0], *outs[0].shape[1:])
        return out.index_copy(0, torch.cat(rows), torch.cat(outs))

    def encode(self, x: torch.Tensor,
               mod_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode the volumes (N, C, H, W, D) of the batch, one per present
        (sample, modality) pair, into tokens (N, T, E). `mod_idx` (N,) is
        required when several encoders or `per_modality_adapter` are set.
        """
        if ((len(self.encoders) > 1 or len(self.adapters) > 1)
                and mod_idx is None):
            raise ValueError(
                "`mod_idx` is required when several encoders or "
                "`per_modality_adapter` are set.")

        # 1. Backbones: one forward per encoder over all its modality slots
        # -> (N, ...) feature vectors or feature maps
        if self.freeze_encoder:
            with torch.no_grad():
                z = self._encode_volumes(x, mod_idx)
        else:
            z = self._encode_volumes(x, mod_idx)

        # 2. Tokenizer(s)
        if len(self.adapters) == 1:      # one shared tokenizer
            return self.adapters[0](z)   # (N, T, E)
        return self._apply_by_group(self.adapters, z, mod_idx)   # (N, T, E)

    def _encode_volumes(self, x: torch.Tensor,
                        mod_idx: Optional[torch.Tensor]) -> torch.Tensor:
        """Volumes (N, C, ...) -> feature maps (N, ...)."""
        if len(self.encoders) == 1:
            return self.encoders[0](x)
        # modality slot -> encoder index (e.g. structural vs diffusion)
        return self._apply_by_group(self.encoders, x,
                                    self.mod_to_encoder[mod_idx])

    def scatter(self, tokens, sample_idx, mod_idx, n_samples):
        """tokens (N, T, D) -> grid (B, M, T, D) + absent mask (B, M)."""
        _, T, D = tokens.shape
        grid = tokens.new_zeros(n_samples, self.num_modalities, T, D)
        grid[sample_idx, mod_idx] = tokens
        absent = torch.ones(n_samples, self.num_modalities,
                            dtype=torch.bool, device=tokens.device)
        absent[sample_idx, mod_idx] = False
        return grid, absent

    def fuse(self, grid, absent, keep_col: Optional[int] = None):
        """`keep_col=None` fuses all present modalities (M*T tokens);
        `keep_col=m` fuses only modality m's T tokens (still absent for
        samples lacking m).
        """
        B, M, T, D = grid.shape
        if keep_col is None:
            x = grid.reshape(B, M * T, D)
            mask = absent.unsqueeze(-1).expand(B, M, T).reshape(B, M * T)
        else:
            x = grid[:, keep_col]                                  # (B, T, D)
            mask = absent[:, keep_col].unsqueeze(-1).expand(B, T)  # (B, T)
        return self.fusion_transformer([x], key_padding_mask=[mask])  # (B, D)

    def forward(self, x, sample_idx, mod_idx, n_samples):
        """Fuse the volumes `x` (N, C, ...) of one augmentation, given the
        sample `sample_idx` (N,) and modality slot `mod_idx` (N,) of each
        entry, and the number of samples `n_samples` (B).

        Returns the M + 1 fused embeddings (B, D), with the prototype (all
        present modalities) last, and the (B, M) modality presence mask.
        """
        grid, absent = self.scatter(self.encode(x, mod_idx), sample_idx,
                                    mod_idx, n_samples)
        z = [self.fuse(grid, absent, keep_col=m)
             for m in range(self.num_modalities)]
        # prototype = all present
        z.append(self.fuse(grid, absent, keep_col=None))
        return z, ~absent


class CoMM(TransformerMixin, BaseEstimator):
    r"""CoMM [1]_ over a single shared encoder or multiple per modality
    encoders, with missing modalities.

    CoMM learns a representation of several modalities by aligning, for two
    augmented views of each sample, the multimodal embedding of one view with
    the embedding of each modality of the other view.

    The model consists of:

    - one or several backbones, encoding the scans of each modality;
    - an adapter, turning the backbone outputs into tokens;
    - a fusion transformer, computing one embedding per modality and one
      multimodal embedding;
    - a projection head, mapping these embeddings to the multimodal space.

    The missing modalities are masked out of the fusion and of the loss.

    Adapted from https://github.com/Duplums/CoMM.

    Parameters
    ----------
    encoder : nn.Module, class of nn.Module or sequence of nn.Module
        Backbone applied to every modality, or several backbones (e.g. one for
        the structural modalities and one for diffusion) dispatched by
        `mod_to_encoder`. A single backbone can also be given as a class,
        instantiated with `encoder_kwargs`. All backbones must output feature
        maps of the same shape.
    num_modalities : int
        Number of modality slots M (matches the modality order of the
        batches).
    encoder_kwargs : dict or None, default=None
        Options for building the encoder. Ignored if `encoder` is
        instantiated.
    mod_to_encoder : sequence of int or None, default=None
        Length-M index of the backbone encoding each modality slot. Required
        when `encoder` holds several backbones.
    adapter : nn.Module or None, default=None
        Tokenizer for the encoder output. Defaults to
        :class:`~nidl.backbones.volume.utils.SimpleFeaturesInputAdapter`
        (pooled vector -> 1 token per modality);
        :class:`~nidl.backbones.volume.utils.Patched3DInputAdapter` tokenizes
        a feature map instead.
    freeze_encoder : bool, default=True
        Freeze the backbone (no grad, eval mode) and train only fusion + head.
    per_modality_adapter : bool, default=False
        Give each modality slot its own copy of `adapter` rather than sharing
        one.
    embed_dim : int, default=2048
        Fusion width. Must equal the adapter's token dim.
    num_heads : int, default=8
        Number of heads in the fusion transformer.
    depth : int, default=1
        Number of attention blocks in the fusion transformer.
    add_bias_kv : bool, default=False
        If True, adds bias to the key and value sequences in the fusion
        transformer.
    attn_drop_rate : float, default=0.0
        Dropout probability on the attention weights of the fusion
        transformer.
    proj_hidden_dim : int, default=512
        Projection head hidden dimension.
    proj_output_dim : int, default=256
        Projection head output dimension.
    temperature : float, default=0.1
        CoMM/InfoNCE temperature.
    subset_weights : sequence of float or None, default=None
        Optional per-subset loss weights (length M+1: one per modality +
        full).
    optimizer : {'sgd', 'adam', 'adamW'} or torch.optim.Optimizer or type, \
        default="adamW"
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
        Trainer options forwarded to `BaseEstimator` (max_epochs, strategy,
        devices, precision, callbacks, logger, ...).

    Attributes
    ----------
    encoder : MissingModMMFusion
        Multi-modal fusion encoder (backbone(s) + adapter + fusion).
    projection_head : CoMMProjectionHead
        Projector that maps the fused embeddings to the latent space.
    loss : CoMMLoss
        CoMM loss, restricted to the observed modalities.

    References
    ----------
    .. [1] Benoit Dufumier, Javiera Castillo-Navarro, et al., "What to align
           in multimodal contrastive learning?", ICLR 2025.
    """

    def __init__(
        self,
        encoder: Union[nn.Module, type[nn.Module], Sequence[nn.Module]],
        num_modalities: int,
        encoder_kwargs: Optional[dict[str, Any]] = None,
        mod_to_encoder: Optional[Sequence[int]] = None,
        adapter: Optional[nn.Module] = None,
        freeze_encoder: bool = True,
        per_modality_adapter: bool = False,
        embed_dim: int = 2048,
        num_heads: int = 8,
        depth: int = 1,
        add_bias_kv: bool = False,
        attn_drop_rate: float = 0.0,
        proj_hidden_dim: int = 512,
        proj_output_dim: int = 256,
        temperature: float = 0.1,
        subset_weights: Optional[Sequence[float]] = None,
        optimizer: Union[str, Optimizer, type] = "adamW",
        learning_rate: float = 3e-4,
        weight_decay: float = 5e-4,
        exclude_bias_and_norm_wd: bool = True,
        optimizer_kwargs: Optional[dict] = None,
        lr_scheduler: Optional[
            Union[str, LRSchedulerPLType]
        ] = "warmup_cosine",
        lr_scheduler_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ):
        ignore = list(kwargs.pop("ignore", []))
        ignore += [name for name in ("callbacks", "encoder", "adapter")
                   if name not in ignore]
        super().__init__(**kwargs, ignore=ignore)

        self.parse_batch = parse_multimodal_batch

        # multi-modal encoder: backbone(s) + tokenizer + latent fusion
        if isinstance(encoder, (list, tuple, nn.ModuleList)):
            encoder = [build_encoder(enc, encoder_kwargs) for enc in encoder]
        else:
            encoder = build_encoder(encoder, encoder_kwargs)
        self._check_sync_batchnorm(encoder, freeze_encoder, kwargs)
        self.encoder = MissingModMMFusion(
            encoder=encoder,
            adapter=(adapter if adapter is not None
                     else SimpleFeaturesInputAdapter()),
            num_modalities=num_modalities,
            mod_to_encoder=mod_to_encoder,
            embed_dim=embed_dim,
            freeze_encoder=freeze_encoder,
            per_modality_adapter=per_modality_adapter,
            num_heads=num_heads,
            depth=depth,
            add_bias_kv=add_bias_kv,
            attn_drop_rate=attn_drop_rate,
        )

        # build a 3-layers projector
        self.projection_head = CoMMProjectionHead(
            input_dim=embed_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim,
        )

        # Build the loss
        self.loss = CoMMLoss(temperature=temperature, weights=subset_weights)

        # optimization config
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.exclude_bias_and_norm_wd = exclude_bias_and_norm_wd
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self._fill_default_lr_scheduler_kwargs()

    @staticmethod
    def _check_sync_batchnorm(encoder, freeze_encoder, trainer_kwargs):
        """Reject `sync_batchnorm=True` with several trainable backbones
        having batch normalization layers.
        """
        encoders = encoder if isinstance(encoder, list) else [encoder]
        if (trainer_kwargs.get("sync_batchnorm", False) and not freeze_encoder
                and len(encoders) > 1
                and any(isinstance(module, nn.modules.batchnorm._BatchNorm)
                        for enc in encoders for module in enc.modules())):
            raise ValueError(
                "`sync_batchnorm=True` is not supported with several "
                "trainable encoders having batch normalization layers: a "
                "device whose batch has no scan for an encoder skips it, "
                "which breaks the synchronization of its batch normalization "
                "layers across devices. The projection head is already "
                "synchronized: remove `sync_batchnorm`, freeze the encoders "
                "or use a single encoder.")

    def forward(self, batch):
        """Encode both augmentations of a multimodal batch.

        The fusion encoder returns one embedding per modality subset (the
        prototype, i.e. all present modalities, last) plus the modality
        presence mask.
        """
        sample_idx, mod_idx = batch["sample_idx"], batch["mod_idx"]
        B = batch["n_samples"]

        # compute features for all modality subsets
        # output structure (k single-modality embeddings + 1 all-modalities
        # embedding)
        z1, present = self.encoder(batch["views"][0], sample_idx, mod_idx, B)
        # all-modalities embedding same as before --> discard it
        z2, _ = self.encoder(batch["views"][1], sample_idx, mod_idx, B)
        z1 = [self.projection_head(z) for z in z1]
        z2 = [self.projection_head(z) for z in z2]
        return z1, z2, present

    def _shared_step(self, batch: dict, is_train: bool = True):
        """Shared code for training and validation steps."""
        z1, z2, present = self(batch)
        # Gather before computing the contrastive loss (the mask without
        # gradients)
        z1 = [gather_tensor(z, module=self, sync_grads=is_train) for z in z1]
        z2 = [gather_tensor(z, module=self, sync_grads=is_train) for z in z2]
        present = gather_tensor(present.float(), module=self).bool()
        return self.loss(z1, z2, present)

    def _log_subsets(self, outputs, stage: str, batch_size: int):
        """Per-subset loss/accuracy: modality slot i, then the prototype last.
        """
        for key, value in outputs.items():
            if key.startswith(("ssl_acc_", "ssl_loss_")):
                self.log(f"{key}/{stage}", value,
                         on_step=False, on_epoch=True, sync_dist=True,
                         batch_size=batch_size)

    def training_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Perform one training step and compute the training loss.

        Parameters
        ----------
        batch: dict or tuple of (dict, torch.Tensor)
            A batch of data from the train dataloader, built with
            :func:`~nidl.utils.data_parsing.collate_multimodal` from samples
            holding two views of each modality.
        batch_idx: int
            The index of the current batch (ignored).
        dataloader_idx: int, default=0
            The index of the dataloader (ignored).

        Returns
        -------
        outputs : dict
            The CoMM loss dict: "loss", "ssl_acc", and the per-subset
            "ssl_loss_i" / "ssl_acc_i".
        """
        batch, _ = self.parse_batch(batch, device=self.device, n_views=2)
        outputs = self._shared_step(batch, is_train=True)
        B = batch["n_samples"]
        self.log("loss/train", outputs["loss"], prog_bar=True, sync_dist=True,
                 batch_size=B)
        self.log("ssl_acc/train", outputs["ssl_acc"], prog_bar=True,
                 sync_dist=True, batch_size=B)
        self.log("loss/train_epoch", outputs["loss"],
                 on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
        self.log("ssl_acc/train_epoch", outputs["ssl_acc"],
                 on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
        self._log_subsets(outputs, "train", B)
        return outputs

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Perform one validation step and compute the validation loss.

        Parameters
        ----------
        batch: dict or tuple of (dict, torch.Tensor)
            A batch of data from the validation dataloader, in the same
            format as in ``training_step``.
        batch_idx: int
            The index of the current batch (ignored).
        dataloader_idx: int, default=0
            The index of the dataloader (ignored).

        Returns
        -------
        outputs : dict
            The CoMM loss dict, as in ``training_step``.
        """
        batch, _ = self.parse_batch(batch, device=self.device, n_views=2)
        outputs = self._shared_step(batch, is_train=False)
        B = batch["n_samples"]
        self.log("loss/val", outputs["loss"], prog_bar=True, sync_dist=True,
                 batch_size=B)
        self.log("ssl_acc/val", outputs["ssl_acc"], prog_bar=True,
                 sync_dist=True, batch_size=B)
        self._log_subsets(outputs, "val", B)
        return outputs

    def test_step(self, batch, batch_idx):
        """Skip the test step."""
        return

    def transform_step(
        self,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Encode the input data into the latent space.
        (discard projection head).

        Parameters
        ----------
        batch: dict or tuple of (dict, torch.Tensor)
            A batch of data, built with
            :func:`~nidl.utils.data_parsing.collate_multimodal` from samples
            holding a single scan of each modality.
        batch_idx: int
            The index of the current batch (ignored).
        dataloader_idx: int, default=0
            The index of the dataloader (ignored).

        Returns
        -------
        features: torch.Tensor
            The multimodal embeddings.
        """
        batch, _ = self.parse_batch(batch, device=self.device)
        z, _ = self.encoder(batch["x"], batch["sample_idx"], batch["mod_idx"],
                            batch["n_samples"])
        return z[-1]        # the prototype: all present modalities fused

    def configure_optimizers(self):
        """Initialize the optimizer and learning rate scheduler in CoMM."""
        backbone_params = [p for p in self.encoder.parameters()
                           if p.requires_grad]
        params = [
            {"name": "backbone", "params": backbone_params},
            {"name": "head", "params": self.projection_head.parameters()},
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
        self.lr_scheduler_kwargs.setdefault("min_lr", 0.0)
