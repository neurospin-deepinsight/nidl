##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from __future__ import annotations

import torch.nn.functional as func
from torch import Tensor

from .dino import DINOLoss


class IBOTPatchLoss(DINOLoss):
    """iBOT masked-patch loss [1]_, as in DINOv2 [2]_.

    DINO cross-entropy at the masked patches, averaged per image, then over
    images. Parameters: see :class:`DINOLoss`.

    References
    ----------
    .. [1] Zhou, J., et al., "iBOT: Image BERT Pre-Training with Online
           Tokenizer." ICLR, 2022. https://arxiv.org/abs/2111.07832
    .. [2] Oquab, M., et al., "DINOv2: Learning Robust Visual Features
           without Supervision." TMLR, 2024. https://arxiv.org/abs/2304.07193
    """

    def forward(
        self,
        teacher_out: Tensor,
        student_out: Tensor,
        masks: Tensor,
        epoch: int | None = None,
    ) -> Tensor:
        """Compute the loss and update the center.

        Parameters
        ----------
        teacher_out: torch.Tensor of shape (n_masked, n_features)
            Teacher outputs at the masked patches.
        student_out: torch.Tensor of shape (n_masked, n_features)
            Student outputs at the masked patches.
        masks: torch.Tensor of shape (n_images, n_patches)
            Boolean masks (True = masked).
        epoch: int or None
            Current epoch, for the teacher temperature.

        Returns
        -------
        loss: torch.Tensor
            The loss.
        """
        if teacher_out.numel() == 0:  # no masked patch
            return student_out.sum() * 0.0
        t_out = self._teacher_probs(teacher_out, epoch)
        s_out = func.log_softmax(student_out / self.student_temp, dim=-1)
        # Each image weighs the same
        weight = 1 / masks.sum(dim=-1, keepdim=True).clamp(min=1)
        weight = weight.expand_as(masks)[masks]
        loss = -((t_out * s_out).sum(dim=-1) * weight).sum() / len(masks)
        self.update_center(teacher_out[None])
        return loss
