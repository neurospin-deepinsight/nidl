##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from collections.abc import Sequence
from typing import Any, Optional, Union

import pytorch_lightning as pl
import torch
import torch.utils.data as data
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.strategies import Strategy
from pytorch_lightning.trainer.connectors.accelerator_connector import (
    _PRECISION_INPUT,
)
from pytorch_lightning.utilities.types import (
    STEP_OUTPUT,
)

from ..utils.validation import _estimator_is, available_if, check_is_fitted

    
class BaseEstimator(pl.LightningModule):
    """ Base class for all estimators in the NIDL framework designed for
    scalability.

    Inherits from PyTorch Lightning's LightningModule.
    This class provides a common interface for training, validation, and
    prediction/transformation in a distributed setting (multi-node multi-GPU)
    inheriting from the Lightning's Trainer capabilities.

    Basicaly, this class defines:

    - a `fit` method.
    - a `transform` or `predict` method if the child class inherit from a
      valid  Mixin class.

    Parameters
    ----------
    callbacks: list of Callback or Callback, default=None
        add a callback or list of callbacks.
    check_val_every_n_epoch: int, default=1
        perform a validation loop after every `N` training epochs. If ``None``,
        validation will be done solely based on the number of training
        batches, requiring ``val_check_interval`` to be an integer value.
    val_check_interval: int or float, default=None
        how often to check the validation set. Pass a ``float`` in the range
        [0.0, 1.0] to check after a fraction of the training epoch. Pass an
        ``int`` to check after a fixed number of training batches. An ``int``
        value can only be higher than the number of training batches when
        ``check_val_every_n_epoch=None``, which validates after every ``N``
        training batches across epochs or during iteration-based training.
        Default: ``1.0``.
    max_epochs: int, default=None
        stop training once this number of epochs is reached. If both
        max_epochs and max_steps are not specified, defaults to
        ``max_epochs = 1000``. To enable infinite training, set
        ``max_epochs = -1``.
    min_epochs: int, default=None
        force training for at least these many epochs.
        Disabled by default.
    max_steps: int, default -1
        stop training after this number of steps. If ``max_steps = -1``
        and ``max_epochs = None``, will default to ``max_epochs = 1000``.
        To enable infinite training, set ``max_epochs`` to ``-1``.
    min_steps: int, default=None
        force training for at least these number of steps.
        Disabled by default.
    enable_checkpointing: bool, default=None
        if ``True``, enable checkpointing. It will configure a default
        ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
        trainer callbacks.
        Default: ``True``.
    enable_progress_bar: bool, default=None
        whether to enable to progress bar by default.
        Default: ``True``.
    enable_model_summary: bool, default=None
        whether to enable model summarization by default.
        Default: ``True``.
    accelerator: str or Accelerator, default="auto"
        supports passing different accelerator types ("cpu", "gpu", "tpu",
        "hpu", "mps", "auto") as well as custom accelerator instances.
    strategy: str or Strategy, default="auto"
        supports different training strategies with aliases as well custom
        strategies.
    devices: listof int, str, int, default="auto"
        the devices to use. Can be set to a positive number (int or str), a
        sequence of device indices (list or str), the value ``-1`` to indicate
        all available devices should be used, or ``"auto"`` for automatic
        selection based on the chosen accelerator.
    num_nodes: int, default=1
        number of GPU nodes for distributed training.
    precision: int or str, default="32-true"
        double precision (64, '64' or '64-true'), full
        precision (32, '32' or '32-true'), 16bit mixed precision (16, '16',
        '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed').
        Can be used on CPU, GPU, TPUs, or HPUs.
    ignore: list of str, default=None
        ignore attribute of instance `nn.Module`.
    random_state: int, default=None
        when shuffling is used, `random_state` affects the ordering of the
        indices, which controls the randomness of each batch. Pass an
        int for reproducible output across multiple function calls.
    kwargs: dict
        lightning's trainer extra parameters.

    Attributes
    ----------
    fitted
        a boolean that is True if the estimator has been fitted, and is False
        otherwise.
    trainer_params
        a dictionaray with the trainer parameters.
    trainer
        the current lightning trainer.
        
    Notes
    -----
    Callbacks can help you to tune, monitor or debug an estimator. For
    instance you can check the type of the input batches using
    `BatchTypingCallback` callback.
    """
    def __init__(
            self,
            callbacks: Optional[Union[list[Callback], Callback]] = None,
            check_val_every_n_epoch: Optional[int] = 1,
            val_check_interval: Optional[Union[int, float]] = None,
            max_epochs: Optional[int] = None,
            min_epochs: Optional[int] = None,
            max_steps: int = -1,
            min_steps: Optional[int] = None,
            enable_checkpointing: Optional[bool] = None,
            enable_progress_bar: Optional[bool] = None,
            enable_model_summary: Optional[bool] = None,
            accelerator: Union[str, Accelerator] = "auto",
            strategy: Union[str, Strategy] = "auto",
            devices: Union[list[int], str, int] = "auto",
            num_nodes: int = 1,
            precision: Optional[_PRECISION_INPUT] = None,
            ignore: Optional[Sequence[str]] = None,
            random_state: Optional[int] = None,
            **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=ignore)
        self.fitted_ = False
        self.trainer_params_ = {
            callbacks: callbacks,
            check_val_every_n_epoch: check_val_every_n_epoch,
            val_check_interval: val_check_interval,
            max_epochs: max_epochs,
            min_epochs: min_epochs,
            max_steps: max_steps,
            min_steps: min_steps,
            enable_checkpointing: enable_checkpointing,
            enable_progress_bar: enable_progress_bar,
            enable_model_summary: enable_model_summary,
            accelerator: accelerator,
            strategy: strategy,
            devices: devices,
            num_nodes: num_nodes,
            precision: precision,
        }
        self.trainer_params_.update(kwargs)

    @property
    def fitted(self):
	    return self.fitted_

    @property
    def trainer_params(self):
	    return self.trainer_params_

    def fit(
            self,
            train_dataloader: data.DataLoader,
            val_dataloader:  Optional[data.DataLoader] = None):
        """ The `fit` method.

        In the child class you will need to define:

        - a `training_step` method for defining the training instructions at
          each step.
        - a `validation_step` method for defining the validation instructions
          at each step.

        Parameters
        ----------
        train_dataloader: torch DataLoader
            training samples.
        val_dataloader: torch DataLoader, default None
            validation samples.

        Returns
        -------
        self: object
            fitted estimator.
        """
        trainer = pl.Trainer(**self.trainer_params_)
        trainer.logger._default_hp_metric = None
        pl.seed_everything(self.hparams.random_state)
        trainer.fit(self, train_dataloader, val_dataloader)
        self.fitted_ = True
        return self

    @available_if(_estimator_is("transformer"))
    def transform(
            self,
            test_dataloader: data.DataLoader):
        """ The `transform` method for transformer.

        In the child class you will need to define:

        - a `transform_step` method for defining the transform instructions at
          each step.

        Parameters
        ----------
        test_dataloader: torch DataLoader
            testing samples.

        Returns
        -------
        out: torch Tensor
            returns transformed samples.
        """
        check_is_fitted(self)
        trainer = pl.Trainer(**self.trainer_params_)
        return torch.cat(trainer.predict(
            self, test_dataloader, return_predictions=True))
    
    @available_if(_estimator_is(("regressor", "classifier", "clusterer")))
    def predict(
            self,
            test_dataloader: data.DataLoader):
        """ The `predict` method for regression, classification and clustering.

        In the child class you will need to define:

        - a `predict_step` method for defining the predict instructions at
          each step.

        Parameters
        ----------
        test_dataloader: torch DataLoader
            testing samples.

        Returns
        -------
        out: torch Tensor
            returns predicted samples.
        """
        check_is_fitted(self)
        trainer = pl.Trainer(**self.trainer_params_)
        return torch.cat(trainer.predict(
            self, test_dataloader, return_predictions=True))

    def training_step(
            self,
            batch: Any,
            batch_idx: int,
            dataloader_idx: Optional[int] = 0) -> STEP_OUTPUT:
        """ Here you compute and return the training loss and some additional
        metrics for e.g. the progress bar or logger.

        Parameters
        ----------
        batch: iterable, normally a :class:`~torch.utils.data.DataLoader`
            the current data.
        batch_idx: int
            the index of this batch.
        dataloader_idx: int, default=0
            the index of the dataloader that produced this batch (only if
            multiple dataloaders are used).

        Returns
        -------
        loss:  STEP_OUTPUT
            the computed loss:

            - :class:`~torch.Tensor` - the loss tensor.
            - ``dict`` - a dictionary which can include any keys, but must
              include the key ``'loss'`` in the case of automatic optimization.
            - ``None`` - in automatic optimization, this will skip to the
              next batch (but is not supported for multi-GPU, TPU, or
              DeepSpeed). For manual optimization, this has no special
              meaning, as returning the loss is not required.

        To use multiple optimizers, you can switch to 'manual optimization'
        and control their stepping:

        Examples
        --------
        >>> def __init__(self):
        >>>     super().__init__()
        >>>     self.automatic_optimization = False
        >>> 
        >>> 
        >>> # Multiple optimizers (e.g.: GANs)
        >>> def training_step(self, batch, batch_idx):
        >>>     opt1, opt2 = self.optimizers()
        >>> 
        >>>     # do training_step with encoder
        >>>     ...
        >>>     opt1.step()
        >>>     # do training_step with decoder
        >>>     ...
        >>>     opt2.step()

        Notes
        -----
        When ``accumulate_grad_batches`` > 1, the loss returned here will be
        automatically normalized by ``accumulate_grad_batches`` internally.
        """
        return super().training_step(batch, batch_idx, dataloader_idx)

    def validation_step(
            self,
            batch: Any,
            batch_idx: int,
            dataloader_idx: Optional[int] = 0) -> STEP_OUTPUT:
        """ Operates on a single batch of data from the validation set. In
        this step you'd might generate examples or calculate anything of
        interest like accuracy.

        Parameters
        ----------
        batch: iterable, normally a :class:`~torch.utils.data.DataLoader`
            the current data.
        batch_idx: int
            the index of this batch.
        dataloader_idx: int, default=0
            the index of the dataloader that produced this batch (only if
            multiple dataloaders are used).

        Returns
        -------
        loss: STEP_OUTPUT
            the computed loss:

            - :class:`~torch.Tensor` - the loss tensor.
            - ``dict`` - a dictionary. can include any keys, but must include
              the key ``'loss'``.
            - ``None`` - skip to the next batch.

        Notes
        -----
        If you don't need to validate you don't need to implement this method.

        Notes
        -----
        When the :meth:`validation_step` is called, the model has been put in
        eval mode and PyTorch gradients have been disabled. At the end of
        validation,  the model goes back to training mode and gradients are
        enabled.
        """
        return super().validation_step(batch, batch_idx, dataloader_idx)

    def transform_step(
            self,
            batch: Any,
            batch_idx: int,
            dataloader_idx: Optional[int] = 0) -> Any:
        """ An alias to :meth:`~nidl.estimators.BaseEstimator.predict_step`.
        """
        return self.predict_step(batch, batch_idx, dataloader_idx)

    def predict_step(
            self,
            batch: Any,
            batch_idx: int,
            dataloader_idx: Optional[int] = 0) -> Any:
        """ Step function called during
        :meth:`~lightning.pytorch.trainer.trainer.Trainer.predict`. By default,
        it calls :meth:`~lightning.pytorch.core.LightningModule.forward`.
        Override to add any processing logic.

        The :meth:`~lightning.pytorch.core.LightningModule.predict_step` is
        used to scale inference on multi-devices.

        To prevent an OOM error, it is possible to use
        :class:`~lightning.pytorch.callbacks.BasePredictionWriter`
        callback to write the predictions to disk or database after each
        batch or on epoch end.

        The :class:`~lightning.pytorch.callbacks.BasePredictionWriter` should
        be used while using a spawn based accelerator. This happens for
        training strategy ``strategy="ddp_spawn"`` or training on 8 TPU cores
        with ``accelerator="tpu", devices=8`` as predictions won't be returned.

        Parameters
        ----------
        batch: iterable, normally a :class:`~torch.utils.data.DataLoader`
            the current data.
        batch_idx: int
            the index of this batch.
        dataloader_idx: int, default=0
            the index of the dataloader that produced this batch (only if
            multiple dataloaders are used).

        Returns
        -------
        out: Any
            the predicted output.
        """
        return super().predict_step(batch, batch_idx, dataloader_idx)


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
