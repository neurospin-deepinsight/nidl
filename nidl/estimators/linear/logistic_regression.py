##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from collections.abc import Sequence
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

from ..base import BaseEstimator, ClassifierMixin


class LogisticRegression(ClassifierMixin, BaseEstimator):
    """ LogisticRegression implementation.

    Parameters
    ----------
    model: nn.Module
        the encoder f(.) architecture.
    num_classes: int
        the number of class to predict.
    lr: float
        the learning rate.
    temperature: float
        the SimCLR loss temperature parameter.
    weight_decay: float
        the Adam optimizer weight decay parameter.
    max_epochs: int, default=None
        optionaly, use a MultiStepLR scheduler.
    random_state: int, default=None
        setting a seed for reproducibility.
    kwargs: dict
        Trainer parameters.

    Attributes
    ----------
    model
        a :class:`~torch.nn.Module` containing the prediction model.
    validation_step_outputs
        a dictionnary with the validation predictions and associated labels
        in the 'pred', and 'label' keys, respectivelly.

    Notes
    -----
    A batch of data must contains two elements: a tensor with images, and a
    tensor with the variable to predict.
    """
    def __init__(
            self,
            model: nn.Module,
            num_classes: int,
            lr: float,
            weight_decay: float,
            random_state: Optional[int] = None,
            **kwargs):
        super().__init__(random_state=random_state, ignore=["model"],
                         **kwargs)
        self.model = model
        self.validation_step_outputs = {}

    def configure_optimizers(self):
        """ Declare a :class:`~torch.optim.AdamW` optimizer and, optionnaly
        (``max_epochs`` is defined), a
        :class:`~torch.optim.lr_scheduler.MultiStepLR` learning-rate
        scheduler.
        """
        optimizer = optim.AdamW(
            self.parameters(), lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay)
        if (hasattr(self.hparams, "max_epochs") and
                self.hparams.max_epochs is not None):
            lr_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[int(self.hparams.max_epochs * 0.6),
                int(self.hparams.max_epochs * 0.8)], gamma=0.1)
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]

    def cross_entropy_loss(
            self,
            batch: tuple[torch.Tensor, Sequence[torch.Tensor]],
            mode: str):
        """ Compute and log the InfoNCE loss using
        :func:`~torch.nn.functional.cross_entropy`.
        """
        imgs, labels = batch
        preds = self.model(imgs)
        loss = func.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        if mode in ("train", "val"):
            self.log(mode + "_loss", loss, prog_bar=True)
            self.log(mode + "_acc", acc, prog_bar=True)
        return preds, loss, labels

    def training_step(
            self,
            batch: tuple[torch.Tensor, Sequence[torch.Tensor]],
            batch_idx: int):
        _, loss, _ = self.cross_entropy_loss(batch, mode="train")
        return loss

    def validation_step(
            self,
            batch: tuple[torch.Tensor, Sequence[torch.Tensor]],
            batch_idx: int):
        preds, loss, labels = self.cross_entropy_loss(batch, mode="val")
        self.validation_step_outputs.setdefault("pred", []).append(preds)
        self.validation_step_outputs.setdefault("label", []).append(labels)

    def on_validation_epoch_end(self):
        """ Clean the validation cache at each epoch ends.
        """
        self.validation_step_outputs.clear()

    def predict_step(
            self,
            batch: Union[tuple[torch.Tensor, Sequence[torch.Tensor]],
                         tuple[torch.Tensor]],
            batch_idx: int):
        imgs = batch[0] 
        return self.model(imgs)
