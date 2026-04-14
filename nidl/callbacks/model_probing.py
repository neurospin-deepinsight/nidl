##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

import numbers
from typing import Union

import pytorch_lightning as pl
from sklearn.base import BaseEstimator as sk_BaseEstimator
from sklearn.metrics import check_scoring
from sklearn.utils.validation import check_array
from torch.utils.data import DataLoader

from nidl.estimators.base import BaseEstimator
from nidl.utils.validation import _estimator_is


class ModelProbingCallback(pl.Callback):
    """Callback to probe the representation of an embedding estimator on a
    dataset.

    It has the following logic:

    1) Embeds the input data (training+test) through the estimator using
       `transform_with_targets` (handles distributed multi-gpu forward pass).
    2) Train the probe on the training embedding (handles multi-cpu training).
    3) Evaluate the probe on the test embedding and log the scores.

    The probing can be performed at the end of training epochs, validation
    epochs, and/or at the start/end of the test epoch.

    The metrics logged depend on the ``scoring`` parameter:

    - If a single score is provided, it logs ``test_score``.
    - If multiple scores are provided, it logs each score with its name
      (such as  ``test_accuracy``, ``test_auc``).

    Eventually, a `prefix_score` can be added to the score names when logging,
    such as ``ridge_`` or ``logreg_`` (giving ``ridge_test_r2`` or
    ``logreg_test_accuracy``).

    Parameters
    ----------
    train_dataloader: torch.utils.data.DataLoader
        Training dataloader yielding batches in the form `(X, y)`
        for further embedding and training of the probe.

    test_dataloader: torch.utils.data.DataLoader
        Test dataloader yielding batches in the form `(X, y)`
        for further embedding and test of the probe.

    probe: sklearn.base.BaseEstimator
        The probe model to be trained on the embedding. It must
        implement `fit` and `predict` methods on numpy array.

    scoring: str, callable, list, tuple, or dict, default=None
        Strategy to evaluate the performance of the `probe` on the test
        set. The scores are logged into the
        :class:`~pytorch_lightning.core.module.LightningModule` during
        training/validation/test according to the configuration of the
        callback.

        If `scoring` represents a single score, one can use:

        - a single string (see :ref:`scoring_string_names`);
        - a callable (see :ref:`scoring_callable`) that returns a single value.
        - `None`, the `probe`'s
          :ref:`default evaluation criterion <scoring_api_overview>` is used.

        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.

    every_n_train_epochs: int or None, default=1
        Number of training epochs after which to run the probing.
        Disabled if None.

    every_n_val_epochs: int or None, default=None
        Number of validation epochs after which to run the probing.
        Disabled if None.

    on_test_epoch_start: bool, default=False
        Whether to run the linear probing at the start of the test epoch.

    on_test_epoch_end: bool, default=False
        Whether to run the linear probing at the end of the test epoch.

    prefix_score: str, default=""
        Prefix to add to the score name when logging. This can be useful when
        using multiple `ModelProbing` callbacks to distinguish the logged
        metrics, such as ``"ridge_"`` or ``"logreg_"``.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from nidl.callbacks import ModelProbingCallback
    >>> callback = ModelProbingCallback(
    ...     train_dataloader=train_loader,
    ...     test_dataloader=test_loader,
    ...     probe=LogisticRegression(),
    ...     scoring=["accuracy", "balanced_accuracy"],
    ...     every_n_train_epochs=5,
    ... )
    """

    def __init__(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        probe: sk_BaseEstimator,
        scoring: Union[str, callable, list, tuple, dict, None] = None,
        every_n_train_epochs: Union[int, None] = 1,
        every_n_val_epochs: Union[int, None] = None,
        on_test_epoch_start: bool = False,
        on_test_epoch_end: bool = False,
        prefix_score: str = "",
    ):
        super().__init__()
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.probe = probe
        self.scoring = scoring
        self.every_n_train_epochs = every_n_train_epochs
        self.every_n_val_epochs = every_n_val_epochs
        self._on_test_epoch_start = on_test_epoch_start
        self._on_test_epoch_end = on_test_epoch_end
        self.prefix_score = prefix_score
        self.counter_val_epochs = 0

        self.scorers = check_scoring(self.probe, scoring=self.scoring)

    def log_metrics(self, trainer, scores):
        """Log the metrics given the predictions and the true labels."""

        if not trainer.is_global_zero:
            return

        # Normalize scores into a dict
        if isinstance(scores, numbers.Number):
            metrics = {f"{self.prefix_score}test_score": float(scores)}

        elif isinstance(scores, dict):
            metrics = {
                f"{self.prefix_score}test_{key}": float(value)
                for key, value in scores.items()
                if isinstance(value, numbers.Number)
            }

        else:
            raise TypeError(
                f"Scores should be a number or a dictionary, got {type(scores)}"
            )

        # Log through all configured loggers
        for logger in trainer.loggers:
            logger.log_metrics(metrics, step=trainer.global_step)

    def probing(self, trainer, pl_module: BaseEstimator):
        """Perform the probing on the given estimator.

        This method performs the following steps:
        1) Extracts the features from the training and test dataloaders
        2) Fits the probe on the training features and labels
        3) Makes predictions on the test features
        4) Computes and logs the metrics.

        Parameters
        ----------
        pl_module: BaseEstimator
            The BaseEstimator module that implements the `transform_step`.

        Raises
        ------
        ValueError: If the pl_module does not inherit from `BaseEstimator` or
        from `TransformerMixin`.

        """
        if not isinstance(pl_module, BaseEstimator) or not _estimator_is(
            "transformer"
        ):
            raise ValueError(
                "Your Lightning module must derive from 'BaseEstimator' and "
                f"'TransformerMixin' got {type(pl_module)}"
            )

        # Embed the data
        X_train, y_train = self.extract_features(
            pl_module, self.train_dataloader
        )
        X_test, y_test = self.extract_features(pl_module, self.test_dataloader)

        # Check arrays
        X_train, y_train = (
            check_array(X_train),
            check_array(y_train, ensure_2d=False),  # can be 1d
        )
        X_test, y_test = (
            check_array(X_test),
            check_array(y_test, ensure_2d=False),  # can be 1d
        )

        # For efficiency, fit/score on rank 0 only
        scores = None
        if trainer.is_global_zero:
            # Fit the probe
            self.probe.fit(X_train, y_train)
            # Compute scores
            scores = self.scorers(self.probe, X_test, y_test)

        if trainer.world_size > 1:
            # Broadcast scores to all ranks
            scores = trainer.strategy.broadcast(scores, src=0)

        # Compute/Log metrics
        self.log_metrics(trainer, scores)

    def extract_features(self, pl_module, dataloader):
        """Extract features from a dataloader with the BaseEstimator.

        By default, it uses the `transform_with_targets` logic to get the
        embeddings with the labels.

        Parameters
        ----------
        trainer: pl.Trainer
            The pytorch-lightning trainer instance.
        pl_module: BaseEstimator
            The BaseEstimator module that implements the 'transform_step'.
        dataloader: torch.utils.data.DataLoader
            The dataloader to extract features from. It should yield batches of
            the form `(X, y)` where `X` is the input data and `y` is the label.

        Returns
        -------
        tuple of (X, y)
            Tuple of numpy arrays (X, y) where X are the extracted features
            and y are the corresponding labels.

        """
        X, y = pl_module.transform_with_targets(dataloader)
        X = X.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        return X, y

    def on_train_epoch_end(self, trainer, pl_module):
        if (
            self.every_n_train_epochs is not None
            and self.every_n_train_epochs > 0
            and trainer.current_epoch % self.every_n_train_epochs == 0
        ):
            self.probing(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if (
            self.every_n_val_epochs is not None
            and self.counter_val_epochs % self.every_n_val_epochs == 0
        ):
            self.probing(trainer, pl_module)

        self.counter_val_epochs += 1

    def on_test_epoch_start(self, trainer, pl_module):
        if self._on_test_epoch_start:
            self.probing(trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if self._on_test_epoch_end:
            self.probing(trainer, pl_module)
