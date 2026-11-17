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
    """Pre-norm transformer block: self-attention, then an MLP, both with
    residual connections.

    Adapted from https://github.com/Duplums/CoMM.

    Parameters
    ----------
    d_model : int
        Dimension of the tokens.
    num_heads : int
        Number of attention heads.
    add_bias_kv : bool, default=False
        Whether to add a learned bias to the keys and values.
    attn_drop_rate : float, default=0.0
        Dropout probability on the attention weights.
    batch_first : bool, default=True
        Whether the tokens are of shape (batch, tokens, features).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        add_bias_kv: bool = False,
        attn_drop_rate: float = 0.0,
        batch_first: bool = True,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            add_bias_kv=add_bias_kv,
            dropout=attn_drop_rate,
            batch_first=batch_first,
        )
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ):
        return self.attn(
            x.clone(),
            x,
            x,
            need_weights=False,
            key_padding_mask=key_padding_mask,
        )[0]

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ):
        x = x + self.attention(self.ln_1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class FusionTransformer(nn.Module):
    """Fusion transformer: self-attention over the concatenated tokens of
    several inputs, summarized by a learned class token.

    Adapted from https://github.com/Duplums/CoMM.

    Parameters
    ----------
    width : int
        Dimension D of the tokens and of the output embedding.
    num_heads : int
        Number of attention heads.
    depth : int
        Number of attention blocks.
    add_bias_kv : bool, default=False
        Whether to add a learned bias to the keys and values.
    attn_drop_rate : float, default=0.0
        Dropout probability on the attention weights.
    """

    def __init__(
        self,
        width: int,
        num_heads: int,
        depth: int,
        add_bias_kv: bool = False,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()

        self.width = width
        self.layers = depth
        self.norm = nn.LayerNorm(width)
        self.cls_token = nn.Parameter(torch.randn(1, 1, width))
        self.resblocks = nn.Sequential(
            *[
                ResidualAttentionBlock(
                    width,
                    num_heads,
                    add_bias_kv=add_bias_kv,
                    attn_drop_rate=attn_drop_rate,
                    batch_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.initialize()

    def initialize(self):
        proj_std = (self.width**-0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.width**-0.5
        fc_std = (2 * self.width) ** -0.5
        for block in self.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(
        self,
        x: list[torch.Tensor],
        key_padding_mask: Optional[list[torch.Tensor]] = None,
    ):
        """Fuse the tokens of several inputs into one embedding.

        Parameters
        ----------
        x : list of torch.Tensor
            The tokens of each input, of shape (B, L_i, D).
        key_padding_mask : list of torch.Tensor or None, default=None
            Boolean masks of shape (B, L_i), one per input, True for the
            tokens to ignore.

        Returns
        -------
        z : torch.Tensor of shape (B, D)
            The output of the class token.
        """
        # Concatenate over tokens + self-attention
        x = torch.cat(x, dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(key_padding_mask, dim=1)
        # append cls token at the beginning
        cls_token = repeat(self.cls_token, "1 1 d -> b 1 d", b=x.shape[0])
        x = torch.cat((cls_token, x), dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                (torch.zeros_like(cls_token[:, :, 0]), key_padding_mask), dim=1
            )

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.masked_fill(
                key_padding_mask.bool(), float("-inf")
            ).float()

        for layer in self.resblocks:
            x = layer(x, key_padding_mask=key_padding_mask)

        x = self.norm(x)
        return x[:, 0]


class MissingModMMFusion(nn.Module):
    """Multimodal encoder of CoMM, with missing modalities.

    It takes the scans of a batch of B samples as one tensor per modality
    :math:`m`, holding the :math:`N_m` samples having it, with the sample of
    each scan. The modalities sharing a backbone and an input shape are
    encoded in a single forward pass. The adapter of each modality then turns
    the backbone outputs into :math:`T_m` tokens of dimension D, where
    :math:`T_m` may differ between modalities. The tokens of each modality
    are scattered into a (B, :math:`T_m`, D) grid, whose missing samples are
    masked out of the fusion transformer.

    Adapted from the MMFusion of https://github.com/Duplums/CoMM.

    Parameters
    ----------
    encoder : nn.Module or sequence of nn.Module
        The backbone, or several backbones dispatched by `mod_to_encoder`.
    adapter : nn.Module or sequence of nn.Module
        Tokenizer of the backbone outputs: a single adapter, shared by all
        the modalities (or copied, see `per_modality_adapter`), or a list of
        M adapters, one per modality.
    num_modalities : int
        Number of modalities M.
    mod_to_encoder : sequence of int or None, default=None
        Index of the backbone encoding each of the M modalities, in the
        order of the batch (the ``modalities`` of `collate_multimodal`).
        Required with several backbones.
    embed_dim : int, default=2048
        Dimension D of the tokens; it must be the token dimension of the
        adapters.
    freeze_encoder : bool, default=False
        Whether to freeze the backbones (no gradient, eval mode).
    per_modality_adapter : bool, default=False
        Whether each modality gets its own copy of a single `adapter`.
    num_heads : int, default=8
        Number of attention heads in the fusion transformer.
    depth : int, default=1
        Number of attention blocks in the fusion transformer.
    add_bias_kv : bool, default=False
        Whether to add a learned bias to the keys and values of the fusion
        transformer.
    attn_drop_rate : float, default=0.0
        Dropout probability on the attention weights of the fusion
        transformer.
    """

    def __init__(
        self,
        encoder: Union[nn.Module, Sequence[nn.Module]],
        adapter: Union[nn.Module, Sequence[nn.Module]],
        num_modalities: int,
        mod_to_encoder: Optional[Sequence[int]] = None,
        embed_dim: int = 2048,
        freeze_encoder: bool = False,
        per_modality_adapter: bool = False,
        num_heads: int = 8,
        depth: int = 1,
        add_bias_kv: bool = False,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"`embed_dim` ({embed_dim}) must be divisible by `num_heads` "
                f"({num_heads})."
            )
        self.embed_dim = embed_dim
        encoders = (
            [encoder] if isinstance(encoder, nn.Module) else list(encoder)
        )
        self.encoders = nn.ModuleList(encoders)
        self.adapters = self._build_adapters(
            adapter, num_modalities, per_modality_adapter
        )
        self.num_modalities = num_modalities
        self.register_buffer(
            "mod_to_encoder",
            self._check_mod_to_encoder(
                mod_to_encoder, num_modalities, len(self.encoders)
            ),
        )
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoders.parameters():
                p.requires_grad = False
            self.encoders.eval()
        self.fusion_transformer = FusionTransformer(
            embed_dim, num_heads, depth, add_bias_kv, attn_drop_rate
        )

    @staticmethod
    def _build_adapters(
        adapter: Union[nn.Module, Sequence[nn.Module]],
        num_modalities: int,
        per_modality_adapter: bool,
    ) -> nn.ModuleList:
        """A single adapter shared by all the modalities, its re-initialized
        copies (`per_modality_adapter`), or the given list of M adapters.
        """
        if not isinstance(adapter, nn.Module):
            adapters = list(adapter)
            if len(adapters) != num_modalities:
                raise ValueError(
                    f"Got {len(adapters)} adapters for {num_modalities} "
                    "modalities: give one adapter per modality."
                )
            if per_modality_adapter:
                raise ValueError(
                    "`per_modality_adapter` copies a single adapter: it "
                    "cannot be used with a list of adapters."
                )
            return nn.ModuleList(adapters)
        if not per_modality_adapter:
            return nn.ModuleList([adapter])
        adapters = nn.ModuleList()
        for _ in range(num_modalities):
            clone = copy.deepcopy(adapter)
            for module in clone.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            adapters.append(clone)
        return adapters

    @staticmethod
    def _check_mod_to_encoder(
        mod_to_encoder: Optional[Sequence[int]],
        num_modalities: int,
        num_encoders: int,
    ) -> torch.Tensor:
        """Modality slot -> encoder index, as a (M,) long tensor."""
        if mod_to_encoder is None:
            if num_encoders > 1:
                raise ValueError(
                    "`mod_to_encoder` is required with several encoders "
                    f"(got {num_encoders})."
                )
            return torch.zeros(num_modalities, dtype=torch.long)
        mod_to_encoder = torch.as_tensor(
            list(mod_to_encoder), dtype=torch.long
        )
        if mod_to_encoder.numel() != num_modalities:
            raise ValueError(
                f"`mod_to_encoder` has {mod_to_encoder.numel()} entries for "
                f"{num_modalities} modalities."
            )
        if (
            int(mod_to_encoder.max()) >= num_encoders
            or int(mod_to_encoder.min()) < 0
        ):
            raise ValueError(
                f"`mod_to_encoder`={mod_to_encoder.tolist()} is out of range "
                f"for {num_encoders} encoder(s)."
            )
        return mod_to_encoder

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:  # keep frozen backbones (incl. BN) in eval
            self.encoders.eval()
        return self

    def _adapter(self, m: int) -> nn.Module:
        """The adapter of modality m."""
        return self.adapters[m if len(self.adapters) > 1 else 0]

    def encode(
        self, xs: Sequence[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        """Encode the scans of a batch into tokens, with the backbone(s) then
        the adapter(s).

        Parameters
        ----------
        xs : list of torch.Tensor or None
            The scans of each of the M modalities, of shape
            (N_m, C_m, ...), or None if no sample of the batch has it.

        Returns
        -------
        tokens : list of torch.Tensor or None
            The tokens of each modality, of shape (N_m, T_m, D), or None if
            no sample of the batch has it.

        Raises
        ------
        ValueError
            If `xs` does not hold M modalities, or if the tokens are not of
            dimension D (`embed_dim`).
        RuntimeError
            If a backbone with synchronized batch normalization is not run
            exactly once (`sync_batchnorm=True`), which would deadlock the
            devices.
        """
        if len(xs) != self.num_modalities:
            raise ValueError(
                f"Got {len(xs)} modalities for {self.num_modalities}: check "
                "the `modalities` given to `collate_multimodal`."
            )

        # 1. Backbones: one forward pass per group of modalities sharing a
        # backbone and an input shape (e.g. a single one for T1w/T2w/FLAIR
        # MRI of the same shape)
        groups = {}
        for m, x in enumerate(xs):
            if x is not None:
                key = (int(self.mod_to_encoder[m]), tuple(x.shape[1:]))
                groups.setdefault(key, []).append(m)
        self._check_sync_batchnorm_calls([e for e, _ in groups])
        feats = [None] * self.num_modalities
        with torch.set_grad_enabled(
            torch.is_grad_enabled() and not self.freeze_encoder
        ):
            for (e, _), mods in groups.items():
                out = self.encoders[e](torch.cat([xs[m] for m in mods]))
                if isinstance(out, tuple):  # e.g. MoE: drop expert scores
                    out = out[0]
                # split the output back into the modalities of the group
                sizes = [len(xs[m]) for m in mods]
                for m, feat in zip(mods, out.split(sizes)):
                    feats[m] = feat

        # 2. Adapters: the T_m tokens of dimension D of each scan
        tokens = [
            None if feat is None else self._adapter(m)(feat)
            for m, feat in enumerate(feats)
        ]
        for m, tok in enumerate(tokens):
            if tok is not None and (
                tok.ndim != 3 or tok.shape[-1] != self.embed_dim
            ):
                raise ValueError(
                    f"The adapter of modality {m} returns tokens of shape "
                    f"{tuple(tok.shape)}, expected (N, T, {self.embed_dim}): "
                    "set `embed_dim` to the token dimension of the adapters "
                    "(e.g. the output dimension of the encoder with the "
                    "default `SimpleFeaturesInputAdapter`)."
                )

        # 3. The backbones and adapters unused by this batch get a zero
        # gradient: in distributed training, every trainable parameter must
        # receive a gradient on every device
        used = {e for e, _ in groups}
        unused = [enc for e, enc in enumerate(self.encoders) if e not in used]
        if len(self.adapters) > 1:
            unused += [
                adapter
                for m, adapter in enumerate(self.adapters)
                if feats[m] is None
            ]
        zero = sum(
            0.0 * p.sum()
            for module in unused
            for p in module.parameters()
            if p.requires_grad
        )
        if isinstance(zero, torch.Tensor):
            m = next(m for m, tok in enumerate(tokens) if tok is not None)
            tokens[m] = tokens[m] + zero
        return tokens

    def _check_sync_batchnorm_calls(self, encoder_calls: list[int]):
        """With synchronized batch normalization, each trainable backbone
        must run exactly once per step on every device, or the devices
        deadlock waiting for each other.
        """
        if self.freeze_encoder or not self.training:
            return
        for e, encoder in enumerate(self.encoders):
            synced = any(
                isinstance(module, nn.SyncBatchNorm)
                for module in encoder.modules()
            )
            if synced and encoder_calls.count(e) != 1:
                raise RuntimeError(
                    f"Backbone {e} has synchronized batch normalization but "
                    f"runs {encoder_calls.count(e)} times in this batch (once "
                    "per input shape of its modalities, none if they are all "
                    "missing), which deadlocks the devices: remove "
                    "`sync_batchnorm`, or give its modalities the same shape."
                )

    def scatter(
        self,
        tokens: Sequence[Optional[torch.Tensor]],
        sample_idx: Sequence[Optional[torch.Tensor]],
        n_samples: int,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Scatter the tokens of each modality into a (B, T_m, D) grid of the
        samples, and return the grids with the (B, M) mask of the missing
        (sample, modality) cells.
        """
        ref = next(tok for tok in tokens if tok is not None)
        absent = torch.ones(
            n_samples,
            self.num_modalities,
            dtype=torch.bool,
            device=ref.device,
        )
        grids = []
        for m, (tok, idx) in enumerate(zip(tokens, sample_idx)):
            if tok is None:  # no sample has m: one masked placeholder token
                grids.append(ref.new_zeros(n_samples, 1, self.embed_dim))
                continue
            grid = tok.new_zeros(n_samples, *tok.shape[1:])
            grid[idx] = tok
            absent[idx, m] = False
            grids.append(grid)
        return grids, absent

    def fuse(
        self,
        grids: Sequence[torch.Tensor],
        absent: torch.Tensor,
        keep_col: Optional[int] = None,
    ) -> torch.Tensor:
        """Fuse the tokens of all the modalities present (`keep_col=None`),
        or of modality `keep_col` only, into embeddings (B, D). A sample
        missing them gets the embedding of the class token alone.
        """
        cols = range(len(grids)) if keep_col is None else [keep_col]
        # the transformer concatenates the tokens of the modalities, and
        # ignores those of the missing cells
        x = [grids[m] for m in cols]
        mask = [absent[:, m, None].expand(-1, grids[m].shape[1]) for m in cols]
        return self.fusion_transformer(x, key_padding_mask=mask)

    def forward(
        self,
        xs: Sequence[Optional[torch.Tensor]],
        sample_idx: Sequence[Optional[torch.Tensor]],
        n_samples: int,
    ):
        """Embed the scans of a batch, one embedding per modality plus the
        multimodal one.

        Parameters
        ----------
        xs : list of torch.Tensor or None
            The scans of each of the M modalities, of shape
            (N_m, C_m, ...), or None if no sample of the batch has it.
        sample_idx : list of torch.Tensor or None
            The sample of each scan of each modality, of shape (N_m,), in
            [0, B), or None if no sample of the batch has it.
        n_samples : int
            The number of samples B in the batch.

        Returns
        -------
        z : list of torch.Tensor
            The M + 1 embeddings, of shape (B, D): one per modality (the
            class token alone for the samples missing it), then the
            multimodal one.
        present : torch.Tensor of shape (B, M)
            Whether each sample has each modality.
        """
        grids, absent = self.scatter(self.encode(xs), sample_idx, n_samples)
        z = [self.fuse(grids, absent, keep_col=m) for m in range(len(grids))]
        # the multimodal embedding: all the modalities present
        z.append(self.fuse(grids, absent, keep_col=None))
        return z, ~absent


class CoMM(TransformerMixin, BaseEstimator):
    r"""CoMM [1]_, with missing modalities.

    CoMM is a multimodal model that learns a representation of several
    modalities with self-supervised contrastive learning. For two augmented
    views of each sample, it computes :math:`M + 1` embeddings: one per
    modality and a multimodal one. Each embedding of one view is then
    aligned with the multimodal embedding of the other view (see
    :class:`~nidl.losses.CoMMLoss`).

    The model consists of:

    - one or several backbones, encoding each modality;
    - adapters, turning the backbone outputs into tokens;
    - a fusion transformer, computing the :math:`M + 1` embeddings from the
      tokens of each modality and their multimodal fusion;
    - a projection head, mapping these embeddings to the space of the loss.

    **CoMM handles missing modalities.** Missing ones are masked out of the
    fusion, of the projection head and of the loss terms: the multimodal
    embedding only fuses the modalities present.

    **Data loader.** The model is trained with batches of samples collated
    by :func:`~nidl.utils.data_parsing.collate_multimodal`. Each sample
    is a dict ``{modality: data or None}``, with ``None`` for a
    missing modality, and ``data`` a list of two augmented views of the scan
    for training, or a single scan at inference.

    After training, the projection head is discarded:
    :meth:`~nidl.estimators.BaseEstimator.transform` returns the multimodal
    embedding of each sample, from the modalities it has.

    Parameters
    ----------
    encoder : nn.Module, class of nn.Module or sequence of nn.Module
        The backbone encoding all the modalities, or several backbones
        dispatched by `mod_to_encoder` (e.g. a CNN for MRI and an MLP for
        tabular data). A single backbone can also be given as a class,
        instantiated with `encoder_kwargs`.
    num_modalities : int
        Number of modalities :math:`M`, i.e. the length of the
        ``modalities`` given to `collate_multimodal`.
    modality_names : sequence of str or None, default=None
        Names of the :math:`M` modalities, in the order of ``modalities``
        (typically the same list, e.g. ``("t1", "t2", "flair")``), used to
        log the loss and accuracy of each term:
        ``ssl_loss_<name>/<stage>`` and ``ssl_acc_<name>/<stage>``, with
        ``fused`` for the multimodal term. By default, ``mod0``, ``mod1``...
    encoder_kwargs : dict or None, default=None
        Options for building the encoder. Ignored if `encoder` is
        instantiated.
    mod_to_encoder : sequence of int or None, default=None
        With several backbones, the index in `encoder` of the backbone
        encoding each of the :math:`M` modalities, in the order of
        ``modalities``: ``mod_to_encoder[m]`` encodes ``modalities[m]``. E.g.
        with ``encoder=[cnn, mlp]`` and ``modalities=("t1", "t2",
        "clinical")``, ``mod_to_encoder=[0, 0, 1]`` encodes the T1w and T2w
        MRI with the CNN, and the clinical data with the MLP.
    adapter : nn.Module, sequence of nn.Module or None, default=None
        Tokenizer of the backbone outputs, shared by all the modalities, or a
        list of :math:`M` adapters, one per modality in the order of
        ``modalities``. By default,
        :class:`~nidl.backbones.volume.utils.SimpleFeaturesInputAdapter`
        turns a feature vector into a single token;
        :class:`~nidl.backbones.volume.utils.Patched3DInputAdapter` tokenizes
        a 3D feature map, and
        :class:`~nidl.backbones.volume.utils.IdentityAdapter` keeps the
        tokens of a transformer.
    freeze_encoder : bool, default=False
        Whether to freeze the backbones (no gradient, eval mode), e.g. when
        they are pretrained: only the adapters, the fusion and the head are
        then trained.
    per_modality_adapter : bool, default=False
        Whether each modality gets its own copy of a single `adapter`,
        instead of sharing it.
    embed_dim : int, default=2048
        Dimension of the tokens and of the embeddings. It must be the token
        dimension of the adapters, i.e. the output dimension of the backbones
        with the default adapter.
    num_heads : int, default=8
        Number of attention heads in the fusion transformer. It must divide
        `embed_dim`.
    depth : int, default=1
        Number of attention blocks in the fusion transformer.
    add_bias_kv : bool, default=False
        Whether to add a learned bias to the keys and values of the fusion
        transformer.
    attn_drop_rate : float, default=0.0
        Dropout probability on the attention weights of the fusion
        transformer.
    proj_hidden_dim : int, default=512
        Projector hidden dimension.
    proj_output_dim : int, default=256
        Projector output dimension.
    temperature : float, default=0.1
        The InfoNCE loss temperature parameter.
    subset_weights : sequence of float or None, default=None
        Weights of the :math:`M + 1` loss terms: one per modality, then the
        multimodal term. By default, all the terms have the same weight.
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
        Additional keyword arguments for the BaseEstimator class, such as
        `max_epochs`, `max_steps`, `devices`, `strategy`, `callbacks`, etc.

    Attributes
    ----------
    encoder : MissingModMMFusion
        The backbones, the adapter(s) and the fusion transformer, mapping the
        scans of each sample to its :math:`M + 1` embeddings.
    projection_head : CoMMProjectionHead
        Projector that maps the embeddings to the space of the loss.
    loss : CoMMLoss
        The CoMM loss function used for training.

    References
    ----------
    .. [1] Benoit Dufumier, Javiera Castillo-Navarro, et al., "What to align
           in multimodal contrastive learning?", ICLR 2025. Official code:
           https://github.com/Duplums/CoMM.
    """

    def __init__(
        self,
        encoder: Union[nn.Module, type[nn.Module], Sequence[nn.Module]],
        num_modalities: int,
        modality_names: Optional[Sequence[str]] = None,
        encoder_kwargs: Optional[dict[str, Any]] = None,
        mod_to_encoder: Optional[Sequence[int]] = None,
        adapter: Optional[Union[nn.Module, Sequence[nn.Module]]] = None,
        freeze_encoder: bool = False,
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
        # Modules cannot be saved as hyper-parameters, classes can (as in
        # SimCLR): keep the encoder classes so that checkpoints reload.
        ignore = list(kwargs.pop("ignore", ["callbacks"]))
        if "callbacks" not in ignore:
            ignore.append("callbacks")
        encoders = encoder if isinstance(encoder, (list, tuple)) else [encoder]
        if (
            any(isinstance(enc, nn.Module) for enc in encoders)
            and "encoder" not in ignore
        ):
            ignore.append("encoder")
        adapters = adapter if isinstance(adapter, (list, tuple)) else [adapter]
        if (
            any(isinstance(adp, nn.Module) for adp in adapters)
            and "adapter" not in ignore
        ):
            ignore.append("adapter")
        super().__init__(**kwargs, ignore=ignore)

        self.parse_batch = parse_multimodal_batch
        self._term_names = self._check_modality_names(
            modality_names, num_modalities
        )

        # multi-modal encoder: backbone(s) + tokenizer + latent fusion
        if isinstance(encoder, (list, tuple, nn.ModuleList)):
            encoder = [build_encoder(enc, encoder_kwargs) for enc in encoder]
        else:
            encoder = build_encoder(encoder, encoder_kwargs)
        self._check_sync_batchnorm(encoder, freeze_encoder, kwargs)
        self.encoder = MissingModMMFusion(
            encoder=encoder,
            adapter=(
                adapter
                if adapter is not None
                else SimpleFeaturesInputAdapter()
            ),
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
    def _check_modality_names(modality_names, num_modalities):
        """Names of the M + 1 loss terms: the modalities, then "fused"."""
        if modality_names is None:
            modality_names = [f"mod{idx}" for idx in range(num_modalities)]
        names = [str(name) for name in modality_names]
        if len(names) != num_modalities:
            raise ValueError(
                f"`modality_names` has {len(names)} names for "
                f"{num_modalities} modalities."
            )
        if len(set(names)) != len(names) or "fused" in names:
            raise ValueError(
                "`modality_names` must be unique and differ from 'fused' "
                f"(the multimodal term), got {names}."
            )
        return [*names, "fused"]

    @staticmethod
    def _check_sync_batchnorm(encoder, freeze_encoder, trainer_kwargs):
        """Reject `sync_batchnorm=True` with several trainable backbones
        having batch normalization layers.
        """
        encoders = encoder if isinstance(encoder, list) else [encoder]
        if (
            trainer_kwargs.get("sync_batchnorm", False)
            and not freeze_encoder
            and len(encoders) > 1
            and any(
                isinstance(module, nn.modules.batchnorm._BatchNorm)
                for enc in encoders
                for module in enc.modules()
            )
        ):
            raise ValueError(
                "`sync_batchnorm=True` is not supported with several "
                "trainable encoders having batch normalization layers: a "
                "device whose batch has no scan for an encoder skips it, "
                "which breaks the synchronization of its batch normalization "
                "layers across devices. Remove `sync_batchnorm`, freeze the "
                "encoders or use a single encoder."
            )

    def forward(self, batch):
        """Embed and project the two views of a batch.

        Parameters
        ----------
        batch : dict
            A parsed batch with the keys "views", "sample_idx" and
            "n_samples", as built by
            :func:`~nidl.utils.data_parsing.collate_multimodal`.

        Returns
        -------
        z1, z2 : list of torch.Tensor
            The :math:`M + 1` projected embeddings of each view, of shape
            (n_samples, proj_output_dim): one per modality (zeros for the
            samples missing it), then the multimodal one.
        present : torch.Tensor of shape (n_samples, M)
            Whether each sample has each modality.
        """
        sample_idx, B = batch["sample_idx"], batch["n_samples"]

        # the M per-modality embeddings, then the multimodal one, per view
        z1, present = self.encoder(batch["views"][0], sample_idx, B)
        z2, _ = self.encoder(batch["views"][1], sample_idx, B)
        present = self._projectable(present)
        z1 = self._project(z1, present)
        z2 = self._project(z2, present)
        return z1, z2, present

    def _projectable(self, present: torch.Tensor) -> torch.Tensor:
        """Drop, during training, the modalities held by a single sample of
        the batch: batch normalization needs at least two samples (unless it
        is synchronized across devices).
        """
        synced = any(
            isinstance(module, nn.SyncBatchNorm)
            for module in self.projection_head.modules()
        )
        if not self.training or synced:
            return present
        return present & (present.sum(dim=0, keepdim=True) > 1)

    def _project(
        self, z: list[torch.Tensor], present: torch.Tensor
    ) -> list[torch.Tensor]:
        """Project the M per-modality embeddings (B, D) on the samples having
        each modality only, so that the placeholder embeddings of the
        missing modalities do not enter the batch normalization of the head
        (they are set to zero and masked out of the loss), then the
        prototype (B, D) on all the samples.
        """
        out = []
        for m, z_m in enumerate(z[:-1]):
            valid = present[:, m]
            proj = self.projection_head(z_m[valid])
            out.append(
                proj.new_zeros(len(z_m), proj.shape[-1]).index_put(
                    (valid,), proj
                )
            )
        out.append(self.projection_head(z[-1]))
        return out

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
        """Per-term loss/accuracy, named after the modalities (the
        multimodal term is "fused"): e.g. ``ssl_loss_t1/train``. Skipped
        terms (fewer than two samples) are not logged.
        """
        for idx, name in enumerate(self._term_names):
            for metric in ("ssl_loss", "ssl_acc"):
                if f"{metric}_{idx}" in outputs:
                    self.log(
                        f"{metric}_{name}/{stage}",
                        outputs[f"{metric}_{idx}"],
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=batch_size,
                    )

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
        self.log(
            "loss/train",
            outputs["loss"],
            prog_bar=True,
            sync_dist=True,
            batch_size=B,
        )
        self.log(
            "ssl_acc/train", outputs["ssl_acc"], sync_dist=True, batch_size=B
        )
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
        self.log(
            "loss/val",
            outputs["loss"],
            prog_bar=True,
            sync_dist=True,
            batch_size=B,
        )
        self.log(
            "ssl_acc/val", outputs["ssl_acc"], sync_dist=True, batch_size=B
        )
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
        """Encode each sample into its multimodal embedding, which fuses the
        modalities it has (the projection head is discarded).

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
        features: torch.Tensor of shape (n_samples, embed_dim)
            The multimodal embeddings.
        """
        batch, _ = self.parse_batch(batch, device=self.device)
        grids, absent = self.encoder.scatter(
            self.encoder.encode(batch["x"]),
            batch["sample_idx"],
            batch["n_samples"],
        )
        # the multimodal embedding only: all the modalities present
        return self.encoder.fuse(grids, absent)

    def configure_optimizers(self):
        """Initialize the optimizer and learning rate scheduler in CoMM."""
        backbone_params = [
            p for p in self.encoder.parameters() if p.requires_grad
        ]
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
