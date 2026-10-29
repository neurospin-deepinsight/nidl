##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Optional, Union

import torch
from torch import nn

from nidl.estimators.ssl.dino import DINO
from nidl.estimators.ssl.utils.momentum import initialize_momentum_params
from nidl.estimators.ssl.utils.projection_heads import DINOProjectionHead
from nidl.losses import IBOTPatchLoss, KoLeoLoss


class DINOv2(DINO):
    """DINOv2 [1]_, for 3D volumes as in 3DINO [2]_.

    :class:`DINO` + an iBOT masked-patch loss (separate head) + a KoLeo
    regularizer on the student CLS features. The encoder must be a
    :class:`~nidl.backbones.volume.VisionTransformer3D` with
    ``mask_token=True`` (and ``dynamic_img_size=True`` for local crops).

    Parameters
    ----------
    encoder : nn.Module or class
        Vision transformer.
    encoder_kwargs : dict or None, default=None
        Options for building the encoder.
    ibot_weight : float, default=1.0
        Weight of the iBOT loss.
    koleo_weight : float, default=0.1
        Weight of the KoLeo loss.
    mask_probability : float, default=0.5
        Fraction of global crops that are masked.
    mask_ratio : tuple of float, default=(0.1, 0.5)
        Range of the masked-patch ratio (3DINO: (0.2, 0.75)).
    **kwargs : dict, optional
        Parameters of :class:`DINO` and of the PL Trainer.

    References
    ----------
    .. [1] Oquab, M., et al., "DINOv2: Learning Robust Visual Features
           without Supervision." TMLR, 2024. https://arxiv.org/abs/2304.07193
    .. [2] Xu, T., et al., "A generalizable 3D framework and model for
           self-supervised learning in medical imaging." npj Digital
           Medicine, 2025. https://doi.org/10.1038/s41746-025-02035-w
    """

    def __init__(
        self,
        encoder: Union[nn.Module, type[nn.Module]],
        encoder_kwargs: Optional[dict[str, Any]] = None,
        ibot_weight: float = 1.0,
        koleo_weight: float = 0.1,
        mask_probability: float = 0.5,
        mask_ratio: tuple[float, float] = (0.1, 0.5),
        **kwargs: Any,
    ):
        super().__init__(encoder, encoder_kwargs, **kwargs)
        if getattr(self.student, "mask_token", None) is None:
            raise ValueError("The encoder needs a mask token.")
        # Separate iBOT heads
        h = self.hparams
        head_kwargs = {
            "input_dim": h.proj_input_dim,
            "hidden_dim": h.proj_hidden_dim,
            "bottleneck_dim": h.proj_bottleneck_dim,
            "output_dim": h.proj_output_dim,
            "batch_norm": h.proj_batch_norm,
            "norm_last_layer": h.proj_norm_last_layer,
        }
        self.student_ibot_head = DINOProjectionHead(**head_kwargs)
        self.teacher_ibot_head = DINOProjectionHead(**head_kwargs)
        initialize_momentum_params(
            self.student_ibot_head, self.teacher_ibot_head
        )
        self.ibot_loss = IBOTPatchLoss(
            output_dim=h.proj_output_dim,
            warmup_teacher_temp=h.warmup_teacher_temp,
            teacher_temp=h.teacher_temperature,
            warmup_teacher_temp_epochs=h.warmup_teacher_temp_epochs,
            student_temp=h.student_temperature,
            centering=h.centering,
        )
        self.koleo_loss = KoLeoLoss()
        self.ibot_weight = ibot_weight
        self.koleo_weight = koleo_weight
        self.mask_probability = mask_probability
        self.mask_ratio = mask_ratio

    def _sample_masks(
        self, n_crops: int, n_patches: int, device: torch.device
    ) -> torch.Tensor:
        """Uniform random masks (n_crops, n_patches): a fraction
        `mask_probability` of crops, ratios spread over `mask_ratio`."""
        n_masked = int(n_crops * self.mask_probability)
        edges = torch.linspace(*self.mask_ratio, n_masked + 1)
        ratios = edges[:-1] + torch.rand(n_masked) * edges.diff()
        n_hidden = torch.zeros(n_crops, dtype=torch.long)
        n_hidden[:n_masked] = (ratios * n_patches).long().clamp(min=1)
        n_hidden = n_hidden[torch.randperm(n_crops)].to(device)
        rank = torch.rand(n_crops, n_patches, device=device).argsort(1)
        return rank.argsort(1) < n_hidden[:, None]

    def _shared_step(self, batch: Sequence[Any]) -> dict[str, Any]:
        """Losses (total, dino, ibot, koleo) and outputs on a batch."""
        X, y = self.parse_batch(
            batch,
            device=self.device,
            num_large_crops=self.num_large_crops,
            num_local_crops=self.num_local_crops,
        )
        glob = torch.cat(X[: self.num_large_crops])
        n_patches = math.prod(self.student._patch_grid(glob))
        masks = self._sample_masks(len(glob), n_patches, glob.device)
        n_prefix = self.student.num_prefix_tokens

        # Student: masked global crops, local crops
        tokens = self.student.forward_features(glob, masks=masks)
        s_cls = self.student.forward_head(tokens)
        s_dino = [self.student_head(s_cls)]
        if self.num_local_crops > 0:
            local = torch.cat(X[self.num_large_crops :])
            s_dino.append(self.student_head(self.student(local)))
        s_dino = torch.cat(s_dino)
        n_tot = self.num_large_crops + self.num_local_crops
        s_dino = s_dino.view(n_tot, -1, s_dino.size(-1))
        s_ibot = self.student_ibot_head(tokens[:, n_prefix:][masks])

        # Teacher: full global crops
        with torch.no_grad():
            tokens = self.teacher.forward_features(glob)
            t_dino = self.teacher_head(self.teacher.forward_head(tokens))
            t_dino = t_dino.view(self.num_large_crops, -1, t_dino.size(-1))
            t_ibot = self.teacher_ibot_head(tokens[:, n_prefix:][masks])

        epoch = self.current_epoch
        dino = self.loss(t_dino, s_dino, epoch=epoch)
        ibot = self.ibot_loss(t_ibot, s_ibot, masks, epoch=epoch)
        koleo = sum(
            self.koleo_loss(c) for c in s_cls.chunk(self.num_large_crops)
        )
        loss = dino + self.ibot_weight * ibot + self.koleo_weight * koleo
        return {
            "loss": loss,
            "dino": dino.detach(),
            "ibot": ibot.detach(),
            "koleo": koleo.detach(),
            "z_student": s_dino.detach(),
            "z_teacher": t_dino.detach(),
            "y": y.detach() if y is not None else None,
        }

    def _log_losses(self, outputs: dict[str, Any], stage: str):
        self.log(
            f"loss/{stage}", outputs["loss"], prog_bar=True, sync_dist=True
        )
        for name in ("dino", "ibot", "koleo"):
            self.log(f"{name}/{stage}", outputs[name], sync_dist=True)

    def training_step(
        self,
        batch: Sequence[Any],
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        """Perform one training step (see :meth:`DINO.training_step`).

        Returns
        -------
        outputs: dict
            As in DINO, plus the loss terms "dino", "ibot" and "koleo".
        """
        outputs = self._shared_step(batch)
        self._log_losses(outputs, "train")
        return outputs

    def validation_step(
        self,
        batch: Sequence[Any],
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        """Perform one validation step (see :meth:`training_step`)."""
        outputs = self._shared_step(batch)
        self._log_losses(outputs, "val")
        return outputs

    def on_train_batch_end(
        self, outputs: dict[str, Any], batch: Sequence[Any], batch_idx: int
    ):
        """Teacher momentum update, iBOT head included."""
        self.momentum_updater.update(
            self.student_ibot_head, self.teacher_ibot_head
        )
        super().on_train_batch_end(outputs, batch, batch_idx)

    def on_after_backward(self):
        """Gradient clipping and last layers freeze."""
        super().on_after_backward()
        if self.current_epoch < self.freeze_last_layer:
            for p in self.student_ibot_head.last_layer.parameters():
                p.grad = None

    def _clipped_modules(self) -> list[nn.Module]:
        """Modules clipped separately: backbone (+ heads)."""
        heads = [self.student_ibot_head] if self.clip_grad_heads else []
        return [*super()._clipped_modules(), *heads]

    def _optim_params(self) -> list[dict[str, Any]]:
        """Parameter groups trained by the optimizer."""
        ibot = {
            "name": "ibot_head",
            "params": self.student_ibot_head.parameters(),
        }
        return [*super()._optim_params(), ibot]
