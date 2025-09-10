##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import logging
from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import LRSchedulerPLType
from torch.optim import Optimizer

from ...losses import KernelMetric, YAwareInfoNCE
from ..base import BaseEstimator, TransformerMixin
from .utils.projection_heads import YAwareProjectionHead


class YAwareContrastiveLearning(TransformerMixin, BaseEstimator):
    """y-Aware Contrastive Learning implementation [1].

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

    References
    ----------
    [1] Contrastive Learning with Continuous Proxy Meta-Data for 3D MRI
        Classification, Dufumier et al., MICCAI 2021

    Parameters
    ----------
    encoder: nn.Module or class
        Which deep architecture to use for encoding the input.
        A PyTorch `torch.nn.Module` is expected.
        In general, the uninstantiated class should be passed, although
        instantiated modules will also work.
    encoder_kwargs: dict or None, default=None
        Options for building the encoder (depends on each architecture).
        Examples:

        - encoder=torchvision.ops.MLP, encoder_kwargs={"in_channels": 10,
          "hidden_channels": [4, 3, 2]} builds an MLP with 3 hidden layers,
          input dim 10, output dim 2.
        - encoder=nidl.volume.backbones.resnet3d.resnet18,
          encoder_kwargs={"n_embedding": 10} builds a ResNet-18 model with
          10 output dimension.

        Ignored if `encoder` is instantiated.
    projection_head: nn.Module or class or None, default=YAwareProjectionHead
        Which projection head to use for the model. If None, no projection head
        is used and the encoder output is directly used for loss computation.
        Otherwise, a `~torch.nn.Module` is expected. In general,
        the uninstantiated class should be passed, although instantiated
        modules will also work. By default, a 2-layer MLP with ReLU activation,
        2048-d hidden units, and 128-d output dimensions is used.
    projection_head_kwargs: dict or None, default=None
        Arguments for building the projection head. By default, input dimension
        is 2048-d and output dimension is 128-d. These can be changed by
        passing a dictionary with keys 'input_dim' and 'output_dim'.
        'input_dim' must be equal to the encoder's output dimension.
        Ignored if `projection_head` is instantiated.
    temperature: float, default=0.1
        Temperature value in y-Aware InfoNCE loss. Small values imply more
        uniformity between samples' embeddings, whereas high values impose
        clustered embedding more sensitive to augmentations.
    kernel: {'gaussian', 'epanechnikov', 'exponential', 'linear', 'cosine'}, \
        default="gaussian"
        Kernel used as a similarity function between auxiliary variables.
    bandwidth: Union[float, int, List[float], array, KernelMetric], \
        default=1.0
        The method used to calculate the bandwidth ("sigma" in [1]) between
        auxiliary variables:

        - If `bandwidth` is a scalar (int or float), it sets the bandwidth to
          a diagnonal matrix with equal values.
        - If `bandwidth` is a 1d array, it sets the bandwidth to a
          diagonal matrix and it must be of size equal to the number of
          features in `y`.
        - If bandwidth is a 2d array, it must be of shape
          `(n_features, n_features)` where `n_features` is the number of
          features in `y`.
        - If `bandwidth` is `KernelMetric`, it uses the `pairwise`
          method to compute the similarity matrix between auxiliary variables.
    optimizer: {'sgd', 'adam', 'adamW'} or torch.optim.Optimizer or type, \
        default="adam"
        Optimizer for training the model. Can be:

        - A string:
        
            - 'sgd': Stochastic Gradient Descent (with optional momentum).
            - 'adam': First-order gradient-based optimizer (default).
            - 'adamW': Adam with decoupled weight decay regularization
              (see "Decoupled Weight Decay Regularization", Loshchilov and
              Hutter, ICLR 2019).
              
        - An instance or subclass of ``torch.optim.Optimizer``.
    optimizer_kwargs: dict or None, default=None
        Arguments for the optimizer ('adam' by default). By default:
        {'betas': (0.9, 0.99), 'weight_decay': 5e-05} where 'betas' are the
        exponential decay rates for first and second moment estimates.

        Ignored if `optimizer` is instantiated.
    learning_rate: float, default=1e-4
        Initial learning rate.
    lr_scheduler: LRSchedulerPLType or class or None, default=None
        Learning rate scheduler to use.
    lr_scheduler_kwargs: dict or None, default=None
        Additional keyword arguments for the scheduler.

        Ignored if `lr_scheduler` is instantiated.
    **kwargs: dict, optional
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
    """
    def __init__(
        self,
        encoder: Union[nn.Module, type[nn.Module]],
        encoder_kwargs: Optional[dict[str, Any]] = None,
        projection_head: Union[
            nn.Module, type[nn.Module], None
        ] = YAwareProjectionHead,
        projection_head_kwargs: Optional[dict[str, Any]] = None,
        temperature: float = 0.1,
        kernel: str = "gaussian",
        bandwidth: Union[float, list[float], np.ndarray, KernelMetric] = 1.0,
        optimizer: Union[str, Optimizer, type[Optimizer]] = "adam",
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        learning_rate: float = 1e-4,
        lr_scheduler: Optional[
            Union[LRSchedulerPLType, type[LRSchedulerPLType]]
        ] = None,
        lr_scheduler_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {"betas": (0.9, 0.99), "weight_decay": 5e-05}
        ignore = []
        if isinstance(encoder, nn.Module):
            ignore.append("encoder")
        if isinstance(projection_head, nn.Module):
            ignore.append("projection_head")
        super().__init__(**kwargs, ignore=ignore)
        self.encoder_kwargs = (
            encoder_kwargs if encoder_kwargs is not None else {}
        )
        self.encoder = self._build_encoder(encoder, self.encoder_kwargs)
        self.projection_head_kwargs = (
            projection_head_kwargs
            if projection_head_kwargs is not None
            else {}
        )
        self.projection_head = self._build_projection_head(
            projection_head, self.projection_head_kwargs
        )
        self.temperature = temperature
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.loss = self._build_loss(
            self.temperature, self.kernel, self.bandwidth
        )
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = (
            lr_scheduler_kwargs if lr_scheduler_kwargs is not None else {}
        )
        self.optimizer_kwargs = optimizer_kwargs

    def training_step(self, batch: Any, batch_idx: int):
        """Perform one training step and computes training loss.

        Parameters
        ----------
        batch: Any
            A batch of data that has been generated from train_dataloader.
            It can be a pair of `torch.Tensor` (V1, V2) or a pair
            ((V1, V2), y) where V1 and V2 are the two views of the same sample
            and y is the auxiliary variable.
        batch_idx: int
            The index of the current batch.

        Returns
        -------
        loss: Tensor
            Training loss computed on this batch of data.
        """
        V1, V2, y = self.parse_batch(batch)
        Z1, Z2 = (
            self.projection_head(self.encoder(V1)),
            self.projection_head(self.encoder(V2)),
        )
        loss = self.loss(Z1, Z2, y)
        self.log("loss/train", loss, prog_bar=True)
        outputs = {
            "loss": loss,
            "Z1": Z1.cpu().detach(),
            "Z2": Z2.cpu().detach(),
            "y_true": y.cpu().detach() if y is not None else None,
        }
        # Returns everything needed for further logging/metrics computation
        return outputs

    def validation_step(self, batch: Any, batch_idx: int):
        """Perform one validation step and computes validation loss.

        Parameters
        ----------
        batch: Any
            A batch of data that has been generated from val_dataloader.
            It can be a pair of `torch.Tensor` (V1, V2) or a pair
            ((V1, V2), y) where V1 and V2 are the two views of the same
            sample and y is the auxiliary variable.
        batch_idx: int
            The index of the current batch.
        """
        V1, V2, y = self.parse_batch(batch)
        Z1, Z2 = (
            self.projection_head(self.encoder(V1)),
            self.projection_head(self.encoder(V2)),
        )
        val_loss = self.loss(Z1, Z2, y)
        outputs = {
            "loss": val_loss,
            "Z1": Z1.cpu().detach(),
            "Z2": Z2.cpu().detach(),
            "y_true": y.cpu().detach() if y is not None else None,
        }
        self.log("loss/val", val_loss, prog_bar=True)
        # Returns everything needed for further logging/metrics computation
        return outputs

    def transform_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        """
        """
        return self.encoder(batch)

    def parse_batch(
        self, batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Parses the batch to extract the two views and the auxiliary
        variable.

        Parameters
        ----------
        batch: Any
            Parse a batch input and return V1, V2, and y.
            The batch can be either:

            - (V1, V2): two views of the same sample.
            - ((V1, V2), y): two views and an auxiliary label or variable.

        Returns
        -------
        V1 : torch.Tensor
            First view of the input.
        V2 : torch.Tensor
            Second view of the input.
        y : Optional[torch.Tensor]
            Auxiliary label or variable, if present. Otherwise, None.
        """
        if isinstance(batch, Sequence) and len(batch) == 2:
            first, second = batch
            if isinstance(first, Sequence) and len(first) == 2:
                # Case: ((V1, V2), y)
                V1, V2 = first
                y = second
            else:
                # Case: (V1, V2)
                V1, V2 = first, second
                y = None
        else:
            raise ValueError(
                "batch should be a pair (V1, V2), or a pair "
                "((V1, V2), y) where V1 and V2 are the two "
                "views of the same sample and y is the auxiliary "
                "variable."
            )
        V1 = V1.to(self.device)
        V2 = V2.to(self.device)
        if y is not None:
            y = y.to(self.device)
        return V1, V2, y

    def configure_optimizers(self):
        """ TODO
        """
        known_optimizers = {
            "adam": torch.optim.Adam,
            "adamW": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
        }
        params = list(self.encoder.parameters()) + list(
            self.projection_head.parameters()
        )
        if isinstance(self.optimizer, str):
            if self.optimizer not in known_optimizers:
                raise ValueError(
                    f"Optimizer '{self.optimizer}' is not implemented. "
                    f"Please use one of the available optimizers: "
                    f"{', '.join(known_optimizers.keys())}"
                )
            optimizer = known_optimizers[self.optimizer](
                params=params, lr=self.learning_rate, **self.optimizer_kwargs
            )
        elif isinstance(self.optimizer, Optimizer):
            if len(self.optimizer_kwargs) > 0:
                logging.getLogger(__name__).warning(
                    "optimizer is already instantiated, ignoring "
                    "'optimizer_kwargs'"
                )
            optimizer = self.optimizer
        elif isinstance(self.optimizer, type) and issubclass(
            self.optimizer, Optimizer
        ):
            optimizer = self.optimizer(
                params=params, lr=self.learning_rate, **self.optimizer_kwargs
            )
        else:
            raise ValueError(
                f"Optimizer must be a string, a PyTorch Optimizer, or a class "
                f"inheriting from Optimizer, got {type(self.optimizer)}"
            )
        if self.lr_scheduler is None:
            scheduler = None
        elif isinstance(self.lr_scheduler, LRSchedulerPLType):
            if len(self.lr_scheduler_kwargs) > 0:
                logging.getLogger(__name__).warning(
                    "lr_scheduler is already instantiated, ignoring "
                    "'lr_scheduler_kwargs'"
                )
            scheduler = self.lr_scheduler
        elif isinstance(self.lr_scheduler, type) and issubclass(
            self.lr_scheduler, LRSchedulerPLType
        ):
            scheduler = self.lr_scheduler(
                optimizer=optimizer, **self.lr_scheduler_kwargs
            )
        if scheduler is None:
            return optimizer
        else:
            return [optimizer], [scheduler]

    def _build_encoder(
        self,
        encoder: Union[str, nn.Module, type[nn.Module]],
        encoder_kwargs: dict[str, Any],
    ) -> nn.Module:
        if isinstance(encoder, nn.Module):
            if encoder_kwargs is not None and len(encoder_kwargs) > 0:
                logging.getLogger(__name__).warning(
                    "encoder is already instantiated, ignoring "
                    "'encoder_kwargs'"
                )
        elif isinstance(encoder, type) and issubclass(encoder, nn.Module):
            encoder = encoder(**encoder_kwargs)
        else:
            raise ValueError(
                f"Encoder must be a string, a PyTorch nn.Module, or a class "
                f"inheriting from nn.Module, got {type(encoder)}"
            )
        return encoder

    def _build_projection_head(
        self,
        projection_head: Union[str, nn.Module, type[nn.Module]],
        projection_head_kwargs: dict[str, Any],
    ) -> nn.Module:
        if projection_head is None:
            projection_head = nn.Identity()
        elif isinstance(projection_head, nn.Module):
            if (
                projection_head_kwargs is not None
                and len(projection_head_kwargs) > 0
            ):
                logging.getLogger(__name__).warning(
                    "projection head is already instantiated, ignoring "
                    "'projection_head_kwargs'"
                )
        elif isinstance(projection_head, type) and issubclass(
            projection_head, nn.Module
        ):
            projection_head = projection_head(**projection_head_kwargs)
        else:
            raise ValueError(
                "Projection head must be None, a string, a PyTorch nn.Module, "
                "or a class inheriting from nn.Module, got "
                f"{type(projection_head)}"
            )
        return projection_head

    def _build_loss(
        self,
        temperature: float,
        kernel: str,
        bandwidth: Union[float, list[float], np.ndarray],
    ) -> nn.Module:
        """Builds the y-Aware InfoNCE loss function with the
        specified temperature, kernel and bandwidth.

        Parameters
        ----------
        temperature: float
            The temperature parameter for the InfoNCE loss.

        kernel: {'gaussian', 'epanechnikov', 'exponential', 'linear', 'cosine'}
            Kernel used as similarity function between auxiliary variables.

        bandwidth: float, list of float, array or KernelMetric
            The method used to calculate the bandwidth in kernel.

        Returns
        -------
        loss: nn.Module
            The y-Aware InfoNCE loss function.
        """
        return YAwareInfoNCE(
            kernel=kernel, bandwidth=bandwidth, temperature=temperature
        )
