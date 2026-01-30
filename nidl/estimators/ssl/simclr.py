##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from ...losses import InfoNCE
from ..base import BaseEstimator, TransformerMixin


class SimCLR(TransformerMixin, BaseEstimator):
    r"""SimCLR [1]_.

    SimCLR is a contrastive learning framework for self-supervised
    representation learning. The key idea is to learn useful features
    without labels by making different augmented views of the same image close
    in a representation space, while pushing apart representations of different
    images. Once trained, the encoder can be reused for downstream tasks such
    as classification or regression.

    The model consists of:

    - A base encoder `f` (e.g., a CNN), which extracts representation vectors.
    - A projection head `g`, which maps representations into a space where the
      contrastive objective is applied.

    During training, two augmented versions of each input are encoded into
    two latent vectors. The objective is to maximize their similarity
    while minimizing the similarity to all other samples in the batch. This is
    achieved with the InfoNCE loss [2]_, [3]_.

    After training, the projection head `g` is discarded, and the encoder `f`
    serves as a pretrained feature extractor. This is because `f` provides
    representations that transfer better to downstream tasks than those from
    `g`.

    Parameters
    ----------
    encoder: nn.Module
        the encoder `f`. It must store the size of the encoded one-dimensional
        feature vector in a `latent_size` parameter.
    hidden_dims: list of str
        the projector `g` with an MLP architecture.
    lr: float
        the learning rate.
    temperature: float
        the SimCLR loss temperature parameter.
    weight_decay: float
        the Adam optimizer weight decay parameter.
    max_epochs: int, default=None
        optionaly, use a CosineAnnealingLR scheduler.
    random_state: int, default=None
        setting a seed for reproducibility.
    kwargs: dict
        Keyword arguments to be passed to the
        :class:`~nidl.estimators.base.BaseEstimator` constructor.

    Attributes
    ----------
    f
        a :class:`~torch.nn.Module` containing the encoder.
    g
        a :class:`~torch.nn.Module` containing the projection head.

    References
    ----------
    .. [1] Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton,
           "A Simple Framework for Contrastive Learning of Visual
           Representations", ICML 2020.
    .. [2] Aaron van den Oord, Yazhe Li, Oriol Vinyals, "Representation
           Learning with Contrastive Predictive Coding", arXiv 2018.
    .. [3] Sohn Kihyuk, "Improved Deep Metric Learning with Multi-class N-pair
           Loss Objective", NIPS 2016.

    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_dims: Sequence[int],
        lr: float,
        temperature: float,
        weight_decay: float,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        ignore = kwargs.pop("ignore", ["encoder", "callbacks"])
        if "encoder" not in ignore:
            ignore.append("encoder")
        if "callbacks" not in ignore:
            ignore.append("callbacks")
        super().__init__(
            random_state=random_state,
            ignore=ignore,
            **kwargs,
        )
        assert temperature > 0.0, "The temperature must be a positive float!"
        assert hasattr(encoder, "latent_size"), (
            "The encoder must store the size of the encoded one-dimensional "
            "feature vector in a `latent_size` parameter!"
        )
        self.f = encoder
        self.g = torchvision.ops.MLP(
            in_channels=self.f.latent_size,
            hidden_channels=hidden_dims,
            activation_layer=nn.ReLU,
            inplace=True,
            bias=True,
            dropout=0.0,
        )
        self.g = nn.Sequential(
            *[
                layer
                for layer in self.g.children()
                if not isinstance(layer, nn.Dropout)
            ]
        )
        self.lr = lr
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.loss = InfoNCE(self.temperature)

    def configure_optimizers(self):
        """Declare a :class:`~torch.optim.AdamW` optimizer and, optionally
        a :class:`~torch.optim.lr_scheduler.CosineAnnealingLR` learning rate
        scheduler if ``max_epochs`` is set.
        """
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        if (
            hasattr(self.hparams, "max_epochs")
            and self.hparams.max_epochs is not None
        ):
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs,
                eta_min=(self.lr / 50),
            )
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        """Perform one training step and computes training loss.

        Parameters
        ----------
        batch: Any
            A batch of data that has been generated from train_dataloader.
            It should be a pair of `torch.Tensor` (V1, V2) where V1 and V2
            are the two views of the same sample. They must have equal first
            dimensions.
        batch_idx: int
            The index of the current batch (ignored).
        dataloader_idx: int, default=0
            The index of the dataloader (ignored).

        Returns
        -------
        loss: Tensor
            Training loss computed on this batch of data.
        """
        V1, V2 = batch[0], batch[1]
        Z1 = self.g(self.f(V1))
        Z2 = self.g(self.f(V2))

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
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        """Perform one validation step and computes validation loss.

        Parameters
        ----------
        batch: Any
            A batch of data that has been generated from val_dataloader.
            It should be a pair of `torch.Tensor` (V1, V2).
        batch_idx: int
            The index of the current batch (ignored).
        dataloader_idx: int, default=0
            The index of the dataloader (ignored).
        """
        V1, V2 = batch[0], batch[1]
        Z1 = self.g(self.f(V1))
        Z2 = self.g(self.f(V2))

        # Gather before computing the contrastive loss.
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
        return self.f(batch)

    def all_gather_and_flatten(self, tensor: torch.Tensor, **kwargs):
        """Gathers the tensor from all devices and flattens batch dimension.

        This is useful when gathering tensors without adding extra dimensions.
        It handles some edge cases, such as when using a single GPU.

        Parameters
        ----------
        tensor: torch.Tensor
            The tensor to gather.
        **kwargs: dict
            Additional keyword arguments for `self.all_gather`.

        Returns
        -------
        tensor: torch.Tensor
            The gathered and flattened tensor.
        """
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
