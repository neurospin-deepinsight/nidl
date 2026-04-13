##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, Union

import torch
from pytorch_lightning.utilities.types import LRSchedulerPLType
from torch import nn
from torch.optim import Optimizer

from ...losses import BarlowTwinsLoss
from ..base import BaseEstimator, TransformerMixin
from .utils.data_parsing import gather_two_views, parse_two_views_batch
from .utils.encoder import build_encoder
from .utils.optimizer import configure_ssl_optimizers
from .utils.projection_heads import BarlowTwinsProjectionHead


class BarlowTwins(TransformerMixin, BaseEstimator):
    """Barlow Twins [1]_.

    Barlow Twins is a self-supervised learning model for learning visual
    representations by i) imposing invariance to data augmentation and
    ii) reducing the redundancy between output features. Contrary to
    contrastive methods, it does **not** rely on negative samples.
    The framework consists of:

    1) Data Augmentation - Generates two augmented views of an image.
    2) Encoder (Backbone Network) - Maps images to feature embeddings
       (e.g., 3D-ResNet).
    3) Projection Head - Maps features to a latent space for Barlow Twins
       loss optimization. The projector dimension in Barlow Twins is typically
       very high (e.g., 8192 or 16384) compared to the features dimension
       (e.g., 2048 in ResNet-50). This is a key difference with other SSL
       methods.
    4) Redundancy reduction loss in addition to a data augmentation invariance
       loss.


    Parameters
    ----------
    encoder : nn.Module or class
        Which deep architecture to use for encoding the input.
        A PyTorch :class:`~torch.nn.Module` is expected.
        In general, the uninstantiated class should be passed, although
        instantiated modules will also work.

    encoder_kwargs : dict or None, default=None
        Options for building the encoder (depends on each architecture).
        Examples:

        - encoder=torchvision.ops.MLP, encoder_kwargs={"in_channels": 10,
          "hidden_channels": [4, 3, 2]} builds an MLP with 3 hidden layers,
          input dim 10, output dim 2.
        - encoder=nidl.volume.backbones.resnet3d.resnet18,
          encoder_kwargs={"n_embedding": 10} builds a ResNet-18 model with
          10 output dimension.

        Ignored if `encoder` is instantiated.

    proj_input_dim : int, default=2048
        Projector input dimension. It must be consistent with encoder's
        output dimension.

    proj_hidden_dim : int, default=2048
        Projector hidden dimension. Original value in [1]_ is 8192, but it
        can be reduced for computational reasons.

    proj_output_dim : int, default=2048
        Projector output dimension. Original value in [1]_ is 8192, but it can
        be reduced for computational reasons.

    lambd : float, default=5e-3
        lambda value in the BarlowTwins loss. Trading off the importance of
        the redundancy reduction term over the invariance term.

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
        `max_epochs`, `max_steps`, `num_sanity_val_steps`,
        `check_val_every_n_epoch`, `callbacks`, etc.

    Attributes
    ----------
    encoder : torch.nn.Module
        Deep neural network mapping input data to low-dimensional vectors.

    projection_head : torch.nn.Module
        Projector that maps encoder output to latent space for loss
        optimization.

    loss : BarlowTwinsLoss
        The BarlowTwins loss function used for training.

    optimizer : torch.optim.Optimizer
        Optimizer used for training.

    lr_scheduler : LRSchedulerPLType or None
        Learning rate scheduler used for training.

    References
    ----------
    .. [1] Zbontar, J., et al., "Barlow Twins: Self-Supervised Learning
           via Redundancy Reduction." PMLR, 2021.
           hhttps://proceedings.mlr.press/v139/zbontar21a
    """

    def __init__(
        self,
        encoder: Union[nn.Module, type[nn.Module]],
        encoder_kwargs: Optional[dict[str, Any]] = None,
        proj_input_dim: int = 2048,
        proj_hidden_dim: int = 2048,
        proj_output_dim: int = 2048,
        lambd: float = 0.005,
        optimizer: Union[str, Optimizer, type[Optimizer]] = "adamW",
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
        # Ignore callbacks and encoder for hyperparameters saving
        ignore = kwargs.pop("ignore", ["callbacks"])
        if "callbacks" not in ignore:
            ignore.append("callbacks")
        if isinstance(encoder, nn.Module) and "encoder" not in ignore:
            ignore.append("encoder")

        super().__init__(**kwargs, ignore=ignore)

        self.parse_batch = parse_two_views_batch
        self.encoder = build_encoder(encoder, encoder_kwargs)

        self.projection_head = BarlowTwinsProjectionHead(
            input_dim=proj_input_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim,
        )

        self.loss = BarlowTwinsLoss(lambd=lambd)
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.exclude_bias_and_norm_wd = exclude_bias_and_norm_wd
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.optimizer_kwargs = optimizer_kwargs

        self._fill_default_lr_scheduler_kwargs()

    def _shared_step(self, batch: Sequence[Any], is_train: bool = True):
        """Shared code for training and validation steps."""
        X, y = self.parse_batch(batch, device=self.device)
        z1 = self.projection_head(self.encoder(X[0]))
        z2 = self.projection_head(self.encoder(X[1]))

        # Gather before computing the contrastive loss.
        z1, z2 = gather_two_views(z1, z2, module=self, sync_grads=is_train)
        loss = self.loss(z1, z2)
        outputs = {
            "loss": loss,
            "z1": z1.detach(),
            "z2": z2.detach(),
            "y": y.detach() if y is not None else None,
        }
        return outputs

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
            A batch of data from the train dataloader. Supported formats are
            ``[X1, X2]`` or ``([X1, X2], y)``, where ``X1`` and ``X2`` are
            tensors representing two augmented views of the same samples.
        batch_idx: int
            The index of the current batch (ignored).
        dataloader_idx: int, default=0
            The index of the dataloader (ignored).

        Returns
        -------
        outputs : dict
            Dictionary containing:
                - "loss": the Barlow Twins loss computed on this batch;
                - "z1": tensor of shape `(batch_size, n_features)`;
                - "z2": tensor of shape `(batch_size, n_features)`;
                - "y": eventual targets (returned as is).
        """
        outputs = self._shared_step(batch, is_train=True)
        self.log("loss/train", outputs["loss"], prog_bar=True, sync_dist=True)
        # Returns everything needed for further logging/metrics computation
        return outputs

    def validation_step(
        self,
        batch: Sequence[Any],
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        """Perform one validation step and computes validation loss.

        Parameters
        ----------
        batch: Sequence[Any]
            A batch of data from the validation dataloader. Supported formats
            are ``[X1, X2]`` or ``([X1, X2], y)``.
        batch_idx: int
            The index of the current batch (ignored).
        dataloader_idx: int, default=0
            The index of the dataloader (ignored).

        Returns
        -------
        outputs : dict
            Dictionary containing:
                - "loss": the Barlow Twins loss computed on this batch;
                - "z1": tensor of shape `(batch_size, n_features)`
                - "z2": tensor of shape `(batch_size, n_features)`
                - "y": eventual targets (returned as is)
        """
        outputs = self._shared_step(batch, is_train=False)
        self.log("loss/val", outputs["loss"], prog_bar=True, sync_dist=True)
        # Returns everything needed for further logging/metrics computation
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
        """Initialize the optimizer and learning rate scheduler in Barlow
        Twins."""
        params = [
            {"name": "backbone", "params": self.encoder.parameters()},
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
