##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from collections.abc import Sequence
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.utils.data as data

from ..callbacks.check_typing import BatchTypingCallback
from ..utils.validation import _estimator_is, available_if, check_is_fitted

    
class BaseEstimator(pl.LightningModule):
    """ Base class for all estimators in nidl.

    Basicaly, this class is a LightningModule with embeded Trainer parameters.
    It defines:

    - a `fit` method.
    - a `transform` or `predict` method if the child class inherit from a
      valid  Mixin class.

    This class also provides a way to connect a custom DataLoader to a
    predefined estimator. Using the `set_batch_connector` method allows you to
    pass a function that reorganizes your batch of data according to the
    estimator's specifications.
    """
    def __init__(
            self,
            random_state: Optional[int] = None,
            ignore: Optional[Sequence[str]] = None,
            hints_batch: Optional[bool] = True,
            **kwargs):
        """ Init class.

        Parameters
        ----------
        random_state: int, default=None
            When shuffling is used, `random_state` affects the ordering of the
            indices, which controls the randomness of each batch. Pass an
            int for reproducible output across multiple function calls.
        ignore: list of str, default=None
            Ignore attribute of instance `nn.Module`.
        hints_batch: bool, default=False
            Require the 'batch' parameter type hints to be specified. This
            type will be checked at runtime using the callback mechanism.
        kwargs: dict
            Trainer parameters.
        """
        super().__init__()
        self.save_hyperparameters(ignore=ignore)
        self.fitted_ = False
        self.trainer_params_ = kwargs
        if hints_batch:
            self.trainer_params_.setdefault("callbacks", []).append(
                BatchTypingCallback()
            )

    def fit(
            self,
            X_train: data.DataLoader,
            X_val:  Optional[data.DataLoader] = None):
        trainer = pl.Trainer(**self.trainer_params_)
        trainer.logger._default_hp_metric = None
        pl.seed_everything(self.hparams.random_state)
        trainer.fit(self, X_train, X_val)
        self.fitted_ = True
        self.best_model_path = trainer.checkpoint_callback.best_model_path
        return self

    @available_if(_estimator_is("transformer"))
    def transform(
            self,
            X_test: data.DataLoader):
        check_is_fitted(self)
        trainer = pl.Trainer(**self.trainer_params_)
        return torch.cat(trainer.predict(
            self, X_test, return_predictions=True))
    
    @available_if(_estimator_is(("regressor", "classifier", "clusterer")))
    def predict(
            self,
            X_test: data.DataLoader):
        check_is_fitted(self)
        trainer = pl.Trainer(**self.trainer_params_)
        return torch.cat(trainer.predict(
            self, X_test, return_predictions=True))
        

class RegressorMixin:
    """ Mixin class for all regression estimators in nidl.

    This mixin sets the estimator type to `"regressor"` through the
    `estimator_type` tag.
    """
    _estimator_type = "regressor"


class ClassifierMixin:
    """ Mixin class for all classifiers in nidl.

    This mixin sets the estimator type to `classifier` through the
    `estimator_type` tag.
    """
    _estimator_type = "classifier"
    
    
class ClusterMixin:
    """ Mixin class for all cluster estimators in nidl.

    This mixin sets the estimator type to `clusterer` through the
    `estimator_type` tag.
    """
    _estimator_type = "clusterer"


class TransformerMixin:
    """ Mixin class for all transformers in nidl.

    This mixin sets the estimator type to `transformer` through the
    `estimator_type` tag.
    """
    _estimator_type = "transformer"   
