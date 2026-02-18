##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from collections.abc import Sequence
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import LRSchedulerPLType
from torch.optim import Optimizer

from ...losses import DINOLoss
from ..base import BaseEstimator, TransformerMixin
from .utils.data_parsing import parse_multi_crops_batch
from .utils.encoder import build_encoder
from .utils.momentum import MomentumUpdater, initialize_momentum_params
from .utils.optimizer import configure_ssl_optimizers
from .utils.projection_heads import DINOProjectionHead


class DINO(TransformerMixin, BaseEstimator):
    """DINO [1]_.

    DINO (self-Distillation with NO labels) is a self-supervised learning
    method for vision models. It learns visual representations using knowledge
    distillation: a student network is trained to align the representation of
    local and global crops (or "views") with the representation of global crops
    given by a teacher model. The teacher is updated through exponential moving
    average of the student, avoiding a representation collapse. The DINO loss
    is a cross-entropy across features between teacher and student
    representations. This way, it does not rely on negative samples as in
    contrastive learning and it is less sensitive to batch size than SimCLR.

    After training, the teacher model is used at inference to obtain image
    features.

    Parameters
    ----------
    encoder : nn.Module or class
        Architecture of the encoder. A PyTorch :class:`~torch.nn.Module`
        is expected. In general, the uninstantiated class should be passed,
        although instantiated modules will also work.
    encoder_kwargs : dict or None, default=None
        Options for building the encoder (depends on each architecture).
        Ignored if `encoder` is already instantiated.
    proj_input_dim : int, default=2048
        Projector input dimension. It must be consistent with encoder's
        output dimension.
    proj_hidden_dim : int, default=2048
        Projector hidden dimension.
    proj_bottleneck_dim : int, default=256
        Projector bottleneck dimension.
    proj_output_dim : int, default=4096
        Projector output dimension.
    proj_batch_norm : bool, default=True
        Whether to use batch norm or not in projector.
        Should be set to False when using a vision transformer backbone.
    proj_norm_last_layer : bool, default=True
        Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the
        training unstable.
    num_local_crops : int, default=8
        Number of local views.
    student_temperature : float, default=0.1
        Temperature for the student.
    teacher_temperature : float, default=0.07
        Final temperature for the teacher.
    warmup_teacher_temp : float, default=0.04
        Initial temperature for the teacher network.
    warmup_teacher_temp_epochs : int, default=30
        Number of epochs for the warmup phase of the teacher temperature.
    base_lambda : float, default=0.996
        Base value for the weighting coefficient in the teacher momentum
        update with exponential moving average. A cosine annealing scheme is
        used.
    final_lambda : float, default=1.0
        Final value for the weighting coefficient in the teacher momentum
        update.
    clip_grad : float, default=0.0
        Threshold for gradient clipping. Null value means no clipping.
    freeze_last_layer : int, default=0
        Number of epochs during which the last layer in student's projection
        head is frozen.
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
    encoder: nn.Module
        Pointer to the teacher.
    student: torch.nn.Module
        Student backbone.
    teacher: torch.nn.Module
        Teacher backbone.
    student_head: torch.nn.Module
        Student head on top of student backbone (only for training).
    teacher_head: torch.nn.Module
        Teacher head on top of teacher backbone (only for training).
    loss: DINOLoss
        The DINO loss used for training.
    optimizer: torch.optim.Optimizer
        Optimizer used for training.
    lr_scheduler: LRSchedulerPLType or None
        Learning rate scheduler used for training.

    Notes
    -----
    We always assume to have 2 global crops (views) in DINO. Adding more views
    becomes computationally prohibitive.


    References
    ----------
    .. [1] Caron, M., et al., "Emerging Properties in Self-Supervised Vision
           Transformers" ICCV, 2021. https://arxiv.org/abs/2104.14294
    """

    def __init__(
        self,
        encoder: Union[nn.Module, type[nn.Module]],
        encoder_kwargs: Optional[dict[str, Any]] = None,
        proj_input_dim: int = 2048,
        proj_hidden_dim: int = 2048,
        proj_bottleneck_dim: int = 256,
        proj_output_dim: int = 4096,
        proj_batch_norm: bool = True,
        proj_norm_last_layer: bool = True,
        num_local_crops: int = 8,
        student_temperature: float = 0.1,
        teacher_temperature: float = 0.07,
        warmup_teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 30,
        base_lambda: float = 0.996,
        final_lambda: float = 1.0,
        clip_grad: float = 0.0,
        freeze_last_layer: int = 0,
        optimizer: Union[str, Optimizer] = "adamW",
        learning_rate: float = 3e-4,
        weight_decay: float = 5e-4,
        exclude_bias_and_norm_wd: bool = True,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        lr_scheduler: Optional[
            Union[str, LRSchedulerPLType]
        ] = "warmup_cosine",
        lr_scheduler_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        ignore = kwargs.pop("ignore", ["callbacks"])
        if "callbacks" not in ignore:
            ignore.append("callbacks")
        if isinstance(encoder, nn.Module) and "encoder" not in ignore:
            ignore.append("encoder")

        super().__init__(**kwargs, ignore=ignore)

        self.parse_batch = parse_multi_crops_batch
        self.student = build_encoder(encoder, encoder_kwargs)
        self.teacher = build_encoder(encoder, encoder_kwargs, deepcopy=True)
        # Copy student params to teacher + no_grad for teacher.
        initialize_momentum_params(self.student, self.teacher)

        self.encoder = self.teacher

        self.student_head = DINOProjectionHead(
            input_dim=proj_input_dim,
            hidden_dim=proj_hidden_dim,
            bottleneck_dim=proj_bottleneck_dim,
            output_dim=proj_output_dim,
            batch_norm=proj_batch_norm,
            norm_last_layer=proj_norm_last_layer,
        )

        self.teacher_head = DINOProjectionHead(
            input_dim=proj_input_dim,
            hidden_dim=proj_hidden_dim,
            bottleneck_dim=proj_bottleneck_dim,
            output_dim=proj_output_dim,
            batch_norm=proj_batch_norm,
            norm_last_layer=proj_norm_last_layer,
        )

        # Copy student head params to teacher head + no_grad for teacher.
        initialize_momentum_params(self.student_head, self.teacher_head)

        self.momentum_updater = MomentumUpdater(base_lambda, final_lambda)

        self.loss = DINOLoss(
            output_dim=proj_output_dim,
            warmup_teacher_temp=warmup_teacher_temp,
            teacher_temp=teacher_temperature,
            warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
            student_temp=student_temperature,
        )

        self.num_local_crops = num_local_crops
        self.num_large_crops = 2  # it is never changed in DINO
        self.clip_grad = clip_grad
        self.freeze_last_layer = freeze_last_layer
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.exclude_bias_and_norm_wd = exclude_bias_and_norm_wd
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.optimizer_kwargs = optimizer_kwargs

        self._fill_default_lr_scheduler_kwargs()

    def training_step(
        self,
        batch: Sequence[Any],
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        """Perform one training step and computes training loss.

        Parameters
        ----------
        batch: Sequence[Any]
            A batch of data in the format [X] or ([X], Y) where [X]
            is a list of torch.Tensor containing `num_large_crops`
            global views (first elements) and `num_small_crops` local
            views (last elements). `Y` are labels (ignored).
        batch_idx: int
            The index of the current batch (ignored).
        dataloader_idx: int, default=0
            The index of the dataloader (ignored).

        Returns
        -------
        outputs: dict
            Dictionary containing:
                - "loss": the DINO loss computed on this batch (scalar);
                - "z_student": tensor of shape
                  `(n_views, batch_size, n_features)`;
                - "z_teacher": tensor of shape
                  `(n_global_views, batch_size, n_features)`;
                - "y": eventual targets (returned as is).
        """
        X, y = self.parse_batch(
            batch,
            device=self.device,
            num_large_crops=self.num_large_crops,
            num_local_crops=self.num_local_crops,
        )
        z_student = self.forward_student(X)
        z_teacher = self.forward_teacher(X)

        loss = self.loss(z_teacher, z_student, epoch=self.current_epoch)
        self.log("loss/train", loss, prog_bar=True, sync_dist=True)
        outputs = {
            "loss": loss,
            "z_student": z_student.detach(),
            "z_teacher": z_teacher.detach(),
            "y": y.detach() if y is not None else None,
        }
        # Returns everything needed for further logging/metrics computation
        return outputs

    def forward_student(self, X: list[torch.Tensor]):
        """Forward global + local views through student."""
        global_views = self.student_head(
            self.student(torch.cat(X[: self.num_large_crops]))
        )
        local_views = self.student_head(
            self.student(torch.cat(X[self.num_large_crops :]))
        )
        # Important: keep the order (global first, local after)
        student_views = torch.cat([global_views, local_views])
        n_tot = self.num_large_crops + self.num_local_crops
        return student_views.view(n_tot, -1, student_views.size(-1))

    @torch.no_grad()
    def forward_teacher(self, X: list[torch.Tensor]):
        """Forward global views only through teacher."""
        global_views = self.teacher_head(
            self.teacher(torch.cat(X[: self.num_large_crops]))
        )
        return global_views.view(
            self.num_large_crops, -1, global_views.size(-1)
        )

    def on_train_batch_end(
        self, outputs: dict[str, Any], batch: Sequence[Any], batch_idx: int
    ):
        """Performs the teacher momentum update.

        Parameters
        ----------
        outputs: dict[str, Any]
            The outputs of the training step (ignored).
        batch: Sequence[Any]
            A batch of input data (ignored).
        batch_idx: int
            The index of the current batch (ignored).
        """

        # update teacher backbone and projector head
        self.momentum_updater.update(self.student, self.teacher)
        self.momentum_updater.update(self.student_head, self.teacher_head)
        # log lambda momentum
        self.log("lambda", self.momentum_updater.cur_lambda)
        # update lambda
        self.momentum_updater.update_lambda(
            cur_step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
        )

    def on_after_backward(self):
        """Performs gradient clipping and last layer freeze if required."""
        # clip gradients
        if self.clip_grad and self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), self.clip_grad
            )
        # zero gradients on last layer
        if self.current_epoch < self.freeze_last_layer:
            for p in self.student_head.last_layer.parameters():
                p.grad = None

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        """Perform one validation step and computes validation loss.

        Parameters
        ----------
        batch: Sequence[Any]
            A batch of data in the format [X] or ([X], Y) where [X]
            is a list of torch.Tensor containing `num_large_crops`
            global views (first elements) and `num_small_crops` local
            views (last elements). `Y` are labels (ignored).
        batch_idx: int
            The index of the current batch (ignored).
        dataloader_idx: int, default=0
            The index of the dataloader (ignored).

        Returns
        -------
        outputs: dict
            Dictionary containing:
                - "loss": the DINO loss computed on this batch (scalar);
                - "z_student": tensor of shape
                  `(n_views, batch_size, n_features)`;
                - "z_teacher": tensor of shape
                  `(n_global_views, batch_size, n_features)`;
                - "y": eventual targets.
        """

        X, y = self.parse_batch(
            batch,
            device=self.device,
            num_large_crops=self.num_large_crops,
            num_local_crops=self.num_local_crops,
        )

        z_student = self.forward_student(X)
        z_teacher = self.forward_teacher(X)

        val_loss = self.loss(z_teacher, z_student, epoch=self.current_epoch)
        self.log("loss/val", val_loss, prog_bar=True, sync_dist=True)
        outputs = {
            "loss": val_loss,
            "z_student": z_student.detach(),
            "z_teacher": z_teacher.detach(),
            "y": y.detach() if y is not None else None,
        }
        # Returns everything needed for further logging/metrics computation
        return outputs

    def test_step(self, batch, batch_idx):
        """Skip the test step."""
        return None

    def transform_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        """Encode the input data into the latent space.

        Importantly, we do not apply the projection head here since it is
        not part of the final model at inference time (only used for training).

        Parameters
        ----------
        batch: torch.Tensor
            A batch of data that has been generated from `test_dataloader`.
            This is given as is to the encoder.
        batch_idx: int
            The index of the current batch (ignored).
        dataloader_idx: int, default=0
            The index of the dataloader (ignored).

        Returns
        -------
        features: torch.Tensor
            The encoded features returned by the encoder.

        """
        return self.encoder(batch)

    def configure_optimizers(self):
        """Initialize the optimizer and learning rate scheduler in DINO."""
        params = [
            {"name": "backbone", "params": self.student.parameters()},
            {"name": "head", "params": self.student_head.parameters()},
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
