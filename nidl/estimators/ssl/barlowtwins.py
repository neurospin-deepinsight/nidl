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

import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import LRSchedulerPLType
from torch.optim import Optimizer

from ...losses import BarlowTwinsLoss
from ..base import BaseEstimator, TransformerMixin
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

    projection_head : nn.Module or class or None,
        default=BarlowTwinsProjectionHead
        Which projection head to use for the model. If None, no projection head
        is used and the encoder output is directly used for loss computation.
        Otherwise, a :class:`~torch.nn.Module` is expected. In general,
        the uninstantiated class should be passed, although instantiated
        modules will also work. By default, a 3-layer MLP with ReLU activation,
        Batch Normalization, 2048-d input dimension, 8192-d hidden units, and
        8192-d output dimensions is used.

    projection_head_kwargs : dict or None, default=None
        Arguments for building the projection head. By default, input dimension
        is 2048-d and output dimension is 8192-d. These can be changed by
        passing a dictionary with keys 'input_dim' and 'output_dim'.
        'input_dim' must be equal to the encoder's output dimension.
        Ignored if `projection_head` is instantiated.

    lambd : float, default=5e-3
        lambda value in the BarlowTwins loss. Trading off the importance of
        the redundancy reduction term vs the data augmentation invariance term.
        In the loss, it is divided by the :math:`D`, the output projector
        dimension.

    optimizer : {'sgd', 'adam', 'adamW'} or torch.optim.Optimizer or type, \
        default="adam"
        Optimizer for training the model. Can be:

        - A string:
        
            - 'sgd': Stochastic Gradient Descent (with optional momentum).
            - 'adam': First-order gradient-based optimizer (default).
            - 'adamW': Adam with decoupled weight decay regularization
              (see "Decoupled Weight Decay Regularization", Loshchilov and
              Hutter, ICLR 2019).
              
        - An instance or subclass of :class:`~torch.optim.Optimizer`.

    optimizer_kwargs : dict or None, default=None
        Arguments for the optimizer ('adam' by default). By default:
        {'betas': (0.9, 0.99), 'weight_decay': 5e-05} where 'betas' are the
        exponential decay rates for first and second moment estimates.

        Ignored if `optimizer` is instantiated.

    learning_rate : float, default=1e-4
        Initial learning rate.

    lr_scheduler : LRSchedulerPLType or class or None, default=None
        Learning rate scheduler to use.

    lr_scheduler_kwargs : dict or None, default=None
        Additional keyword arguments for the scheduler.

        Ignored if `lr_scheduler` is instantiated.

    **kwargs : dict, optional
        Additional keyword arguments for the BaseEstimator class, such as
        `max_epochs`, `max_steps`, `num_sanity_val_steps`,
        `check_val_every_n_epoch`, `callbacks`, etc.

    Attributes
    ----------
    encoder : torch.nn.Module
        Deep neural network mapping input data to low-dimensional vectors.

    projection_head : torch.nn.Module
        Maps encoder output to latent space for contrastive loss optimization.

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
        projection_head: Union[
            nn.Module, type[nn.Module], None
        ] = BarlowTwinsProjectionHead,
        projection_head_kwargs: Optional[dict[str, Any]] = None,
        lambd: float = 0.005,
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
        ignore = ["callbacks"]
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
        self.lambd = lambd
        self.loss = self._build_loss(self.lambd)
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = (
            lr_scheduler_kwargs if lr_scheduler_kwargs is not None else {}
        )
        self.optimizer_kwargs = optimizer_kwargs

    def training_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        """Perform one training step and computes training loss.

        Parameters
        ----------
        batch: Any
            A batch of data that has been generated from train_dataloader.
            It is a pair of `torch.Tensor` (V1, V2).
        batch_idx: int
            The index of the current batch (ignored).
        dataloader_idx: int, default=0
            The index of the dataloader (ignored).

        Returns
        -------
        loss: Tensor
            Training loss computed on this batch of data.
        """
        V1, V2 = self.parse_batch(batch)
        Z1, Z2 = (
            self.projection_head(self.encoder(V1)),
            self.projection_head(self.encoder(V2)),
        )
        # Gather before computing the contrastive loss.
        Z1 = self.all_gather_and_flatten(Z1, sync_grads=True)
        Z2 = self.all_gather_and_flatten(Z2, sync_grads=True)

        loss = self.loss(Z1, Z2)
        self.log("loss/train", loss, prog_bar=True, sync_dist=True)
        outputs = {
            "loss": loss,
            "Z1": Z1.cpu().detach(),
            "Z2": Z2.cpu().detach(),
        }
        # Returns everything needed for further logging/metrics computation
        return outputs

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        """Perform one validation step and computes validation loss.

        Parameters
        ----------
        batch: Any
            A batch of data that has been generated from val_dataloader.
            It is a pair of `torch.Tensor` (V1, V2)
            where V1 and V2 are the two views of the same
            sample.
        batch_idx: int
            The index of the current batch (ignored).
        dataloader_idx: int, default=0
            The index of the dataloader (ignored).
        """

        V1, V2 = self.parse_batch(batch)
        Z1, Z2 = (
            self.projection_head(self.encoder(V1)),
            self.projection_head(self.encoder(V2)),
        )
        Z1 = self.all_gather_and_flatten(Z1, sync_grads=False)
        Z2 = self.all_gather_and_flatten(Z2, sync_grads=False)

        val_loss = self.loss(Z1, Z2)
        outputs = {
            "loss": val_loss,
            "Z1": Z1.cpu().detach(),
            "Z2": Z2.cpu().detach(),
        }
        self.log("loss/val", val_loss, prog_bar=True, sync_dist=True)
        # Returns everything needed for further logging/metrics computation
        return outputs

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

    def parse_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Parses the batch to extract the two views and the auxiliary
        variable.

        Parameters
        ----------
        batch: Any
            Parse a batch input and return V1, V2.
            The batch is:

            - (V1, V2): two views of the same sample.

        Returns
        -------
        V1 : torch.Tensor
            First view of the input.
        V2 : torch.Tensor
            Second view of the input.

        """
        if isinstance(batch, Sequence) and len(batch) == 2:
            first, second = batch
            V1, V2 = first, second
        else:
            raise ValueError(
                "batch should be a pair (V1, V2)"
                "where V1 and V2 are the two "
                "views of the same sample"
            )
        V1 = V1.to(self.device)
        V2 = V2.to(self.device)
        return V1, V2

    def configure_optimizers(self):
        """Instantiate the required optimizer and setup the scheduler."""
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

    def all_gather_and_flatten(
        self, tensor: Union[torch.Tensor, None], **kwargs
    ):
        """Gathers the tensor from all devices and flattens batch dimension.

        This is useful when gathering tensors without adding extra dimensions.
        It handles some edge cases, such as when using a single GPU.

        Parameters
        ----------
        tensor: torch.Tensor or None
            The tensor to gather. If None, it is returned as is.
        **kwargs: dict
            Additional keyword arguments for `self.all_gather`.

        Returns
        -------
        tensor: torch.Tensor
            The gathered and flattened tensor.
        """
        if tensor is None:
            return tensor
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"tensor must be a torch.Tensor, got {type(tensor)}"
            )
        if self.trainer is None or self.trainer.world_size == 1:
            return tensor
        gathered = self.all_gather(tensor, **kwargs)
        # Reshape to (batch_size * world_size, *)
        gathered = gathered.reshape(-1, *gathered.shape[2:])
        return gathered

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
        lambd: float,
    ) -> nn.Module:
        """Builds the Barlow Twins loss object with the specified lambda
        parameter.

        Parameters
        ----------
        temperature: float
            The lambda parameter for the BarlowTwins loss.

        Returns
        -------
        loss: nn.Module
            The BarlowTwins loss function.
        """
        return BarlowTwinsLoss(lambd=lambd)
