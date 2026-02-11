##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import LRSchedulerPLType
from torch.optim import Optimizer

from ...losses import KernelMetric, YAwareInfoNCE
from ..base import BaseEstimator, TransformerMixin
from .utils.data_parsing import (
    gather_tensor,
    gather_two_views,
    parse_two_views_batch,
)
from .utils.encoder import build_encoder
from .utils.optimizer import configure_ssl_optimizers
from .utils.projection_heads import YAwareProjectionHead


class YAwareContrastiveLearning(TransformerMixin, BaseEstimator):
    """y-Aware Contrastive Learning [1]_.

    y-Aware Contrastive Learning is a self-supervised learning framework for
    learning visual representations with auxiliary variables. It leverages
    contrastive learning by maximizing the agreement between differently
    augmented views of images with similar auxiliary variables while minimizing
    agreement between different images. The framework consists of:

    1) Data Augmentation - Generates two augmented views of an image.
    2) Kernel - Similarity function between auxiliary variables.
    3) Encoder (Backbone Network) - Maps images to feature embeddings
       (e.g., 3D-ResNet).
    4) Projection Head - Maps features to a latent space for contrastive
       loss optimization.
    5) Contrastive Loss (y-Aware) - Encourages augmented views of
       i) the same image and ii) images with close auxiliary variables
       to be closer while pushing dissimilar ones apart.


    Parameters
    ----------
    encoder : nn.Module or class
        Which deep architecture to use for encoding the input.
        A PyTorch `torch.nn.Module` is expected.
        In general, the uninstantiated class should be passed, although
        instantiated modules will also work.
    encoder_kwargs : dict or None, default=None
        Options for building the encoder (depends on each architecture).
        Ignored if `encoder` is instantiated.
    proj_input_dim : int, default=2048
        Projector input dimension. It must be consistent with encoder's
        output dimension.
    proj_hidden_dim : int, default=512
        Projector hidden dimension.
    proj_output_dim : int, default=128
        Projector output dimension.
    temperature : float, default=0.1
        Temperature value in y-Aware InfoNCE loss. Small values imply more
        uniformity between samples' embeddings, whereas high values impose
        clustered embedding more sensitive to augmentations.
    kernel : {'gaussian', 'epanechnikov', 'exponential', 'linear', 'cosine'}, \
        default="gaussian"
        Kernel used as a similarity function between auxiliary variables.
    bandwidth : Union[float, List[float], array, KernelMetric], \
        default=1.0
        The method used to calculate the bandwidth ("sigma" in [1]) between
        auxiliary variables:

        - If `bandwidth` is a scalar, it sets the bandwidth to a diagnonal
          matrix with equal values.
        - If `bandwidth` is a 1d array, it sets the bandwidth to a
          diagonal matrix and it must be of size equal to the number of
          features in `y`.
        - If `bandwidth` is a 2d array, it must be of shape
          `(n_features, n_features)` where `n_features` is the number of
          features in `y`.
        - If `bandwidth` is `KernelMetric`, it uses the `pairwise`
          method to compute the similarity matrix between auxiliary variables.
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
    encoder: torch.nn.Module
        Deep neural network mapping input data to low-dimensional vectors.
    projection_head: torch.nn.Module
        Maps encoder output to latent space for contrastive loss optimization.
    loss: yAwareInfoNCE
        The yAwareInfoNCE loss function used for training.
    optimizer: torch.optim.Optimizer
        Optimizer used for training.
    lr_scheduler: LRSchedulerPLType or None
        Learning rate scheduler used for training.

    References
    ----------
    .. [1] Dufumier, B., et al., "Contrastive learning with continuous proxy
           meta-data for 3D MRI classification." MICCAI, 2021.
           https://arxiv.org/abs/2106.08808
    """

    def __init__(
        self,
        encoder: Union[nn.Module, type[nn.Module]],
        encoder_kwargs: Optional[dict[str, Any]] = None,
        proj_input_dim: int = 2048,
        proj_hidden_dim: int = 512,
        proj_output_dim: int = 128,
        temperature: float = 0.1,
        kernel: str = "gaussian",
        bandwidth: Union[float, list[float], np.ndarray, KernelMetric] = 1.0,
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
        ignore = kwargs.pop("ignore", ["callbacks"])
        if "callbacks" not in ignore:
            ignore.append("callbacks")
        if isinstance(encoder, nn.Module) and "encoder" not in ignore:
            ignore.append("encoder")

        super().__init__(
            ignore=ignore,
            **kwargs,
        )

        self.parse_batch = parse_two_views_batch
        self.encoder = build_encoder(encoder, encoder_kwargs)

        self.projection_head = YAwareProjectionHead(
            input_dim=proj_input_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim,
        )

        self.temperature = temperature
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.exclude_bias_and_norm_wd = exclude_bias_and_norm_wd
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self._fill_default_lr_scheduler_kwargs()

        self.loss = YAwareInfoNCE(
            self.kernel, self.bandwidth, self.temperature
        )

    def _shared_step(self, batch: Sequence[Any], is_train: bool = True):
        """Shared code for training and validation steps."""
        X, y = self.parse_batch(batch, device=self.device)
        z1 = self.projection_head(self.encoder(X[0]))
        z2 = self.projection_head(self.encoder(X[1]))

        # Gather before computing the contrastive loss.
        z1, z2 = gather_two_views(
            z1, z2, trainer=self.trainer, sync_grads=is_train
        )
        y = (
            gather_tensor(y, trainer=self.trainer, sync_grads=is_train)
            if y is not None
            else None
        )
        loss = self.loss(z1, z2, y)
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
        dataloader_idx: int = 0,
    ):
        """Perform one training step and computes training loss.

        Parameters
        ----------
        batch: Sequence[Any]
            A batch of data from the train dataloader. Supported formats are
            ``[X1, X2]`` or ``([X1, X2], y)``, where ``X1`` and ``X2`` are
            tensors representing two augmented views of the same samples and
            ``y`` is the auxiliary variable (e.g., age).
        batch_idx: int
            The index of the current batch (ignored).
        dataloader_idx: int, default=0
            The index of the dataloader (ignored).

        Returns
        -------
        outputs : dict
            Dictionary containing:
                - "loss": the y-Aware loss computed on this batch;
                - "z1": tensor of shape `(batch_size, n_features)`;
                - "z2": tensor of shape `(batch_size, n_features)`;
                - "y": auxiliary variables.
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
                - "loss": the y-Aware loss computed on this batch;
                - "z1": tensor of shape `(batch_size, n_features)`;
                - "z2": tensor of shape `(batch_size, n_features)`;
                - "y": auxiliary variables.
        """
        outputs = self._shared_step(batch, is_train=False)
        self.log("loss/val", outputs["loss"], prog_bar=True, sync_dist=True)
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
        """Initialize the optimizer and learning rate scheduler in y-Aware."""
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
        self.lr_scheduler_kwargs.setdefault("min_lr", 0)
