##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from copy import deepcopy
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.data as data
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from nidl.utils import Bunch

from ..estimators.base import BaseEstimator, TransformerMixin


class CachingCallback(pl.Callback):
    """ Cache embeddings.

    This caching callback can be usefull in many situations such as model
    probing.

    Parameters
    ----------
    train_dataloader: torch.utils.data.DataLoader
        training dataloader.
    validation_dataloader: torch.utils.data.DataLoader, default=None
        validation dataloader.
    test_dataloader: torch.utils.data.DataLoader, default=None
        test dataloader.
    auxiliaries: object, defautl=None
        some auxiliary variables that will be shared.
    frequency: int, default=1
        when to update the caching, by default after each validation loop.
    model_attr: str, default='model'
        the attribute name on your estimator containg the model to inspect.
        For recursion, use the '.' as a separator. For example if you only
        want to access the model encoder you may specify 'model.encoder'.

    Attributes
    ----------
    embeddings: nidl.utils.Bunch
        contains the the different latent representations in the `train`,
        `validation` and `test` keys as well as the last upadate epoch number
        in `last_epoch`. The input auxiliaries are also available in
        `auxiliaries`. This attributes is made available from the
        input estimator.

    Notes
    -----
    The caching is performed during the
    :meth:`~CachingCallback.on_validation_epoch_end` event with the
    specified input `frequency`. Thus, if no validation data loader is
    specified in the `fit` this callback is not used.

    Raises
    ------
    TypeError
        if the input module is an estimator that do not dervies from a valid
        mixin type.
    AttributeError
        when the `model_attr` layer is missing or when one data loader
        is shuffled (for this we check if the name of the sampler contains
        the 'random' string).
    """
    def __init__(
            self,
            train_dataloader: DataLoader,
            validation_dataloader: Optional[DataLoader] = None,
            test_dataloader: Optional[DataLoader] = None,
            auxiliaries: Optional[Any] = None,
            frequency: int = 1,
            model_attr: str = "model"):
        super().__init__()
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        for loader in ():
            name = loader.sampler.__class__.__name__
            if "random" in name.lower():
                raise AttributeError(
                    f"From the data loader name '{name}' we detected that it "
                    "may be shuffled."
                )
        self.frequency = frequency
        self.model_attr = model_attr
        self.embeddings = Bunch(auxiliaries=auxiliaries)

    def get_module(
            self,
            pl_module: LightningModule) -> nn.Module:
        module = pl_module
        for attr in self.model_attr.split("."):
            module = getattr(module, attr)
        return module

    def on_validation_epoch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule) -> None:
        if trainer.sanity_checking:  # optional skip
            return
        if pl_module.current_epoch % self.frequency != 0:
            return
        self.embeddings["last_epoch"] = pl_module.current_epoch
        model = self.get_module(pl_module)
        if model is None:
            raise AttributeError(
                f"The {pl_module.__class__.__name__} estimator has no "
                f"'{self.model_attr}' attribute."
            )
        params = deepcopy(pl_module.trainer_params)
        params["barebones"] = True
        params["callbacks"] = []
        tmp_pl_module = CachingEstimator(model, **params)
        for name, loader in [("train", self.train_dataloader),
                             ("validation", self.validation_dataloader),
                             ("test", self.test_dataloader)]:
            if loader is None:
                self.embeddings[name] = None
            else:
                self.embeddings[name] = tmp_pl_module.transform(loader)
        pl_module.embeddings = self.embeddings


class CachingEstimator(TransformerMixin, BaseEstimator):
    """ Dummy estimator to perform inference on an isolated environement.

    Parameters
    ----------
    model: nn.Module
        the inference model.
    kwargs: dict
        trainer parameters.
    """
    def __init__(
            self,
            model: nn.Module,
            **kwargs):
        super().__init__(ignore=["model"], **kwargs)
        self.model = model

    def transform_step(
            self,
            batch: torch.Tensor,
            batch_idx: int,
            dataloader_idx: Optional[int] = 0):
        return self.model(batch)

    def transform(
            self,
            test_dataloader: data.DataLoader) -> Any:
        trainer = pl.Trainer(**self.trainer_params_)
        return torch.cat(trainer.predict(
            self, test_dataloader, return_predictions=True))
