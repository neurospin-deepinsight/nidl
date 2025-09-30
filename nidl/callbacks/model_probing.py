##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from abc import ABC, abstractmethod
from typing import Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_only
from sklearn.base import BaseEstimator as sk_BaseEstimator
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.utils.validation import check_array
from torch.utils.data import DataLoader, DistributedSampler

from nidl.estimators.base import BaseEstimator
from nidl.metrics import regression_report
from nidl.utils.validation import _estimator_is


class ModelProbing(ABC, pl.Callback):
    """Callback to probe the representation of an embedding estimator on a
    dataset.

    It has the following logic:

    1) Embeds the input data (training+test) through the estimator using
       `transform_step` method (handles distributed multi-gpu forward pass).
    2) Train the probe on the training embedding (handles multi-cpu training).
    3) Test the probe on the test embedding and log the metrics.

    This callback is abstract and should be inherited to implement
    the `log_metrics` method.

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
    prog_bar: bool, default=True
        Whether to display the metrics in the progress bar.
    """

    def __init__(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        probe: sk_BaseEstimator,
        every_n_train_epochs: Union[int, None] = 1,
        every_n_val_epochs: Union[int, None] = None,
        on_test_epoch_start: bool = False,
        on_test_epoch_end: bool = False,
        prog_bar: bool = True,
    ):
        super().__init__()
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.probe = probe
        self.every_n_train_epochs = every_n_train_epochs
        self.every_n_val_epochs = every_n_val_epochs
        self._on_test_epoch_start = on_test_epoch_start
        self._on_test_epoch_end = on_test_epoch_end
        self.prog_bar = prog_bar
        self.counter_train_epochs = 0
        self.counter_val_epochs = 0

    @rank_zero_only
    def fit(self, X, y):
        """Fit the probe on the training data embeddings."""
        return self.probe.fit(X, y)

    @rank_zero_only
    def predict(self, X):
        """Make predictions on new data."""
        return self.probe.predict(X)

    @abstractmethod
    @rank_zero_only
    def log_metrics(self, pl_module, y_pred, y_true):
        """Log the metrics given the predictions and the true labels."""

    @staticmethod
    def adapt_dataloader_for_ddp(dataloader, trainer):
        """Wrap user dataloader with DistributedSampler if in DDP mode."""
        dataset = dataloader.dataset

        if trainer.world_size > 1:
            # Create a distributed sampler
            sampler = DistributedSampler(
                dataset,
                num_replicas=trainer.world_size,
                rank=trainer.global_rank,
                shuffle=False,
            )
            # Recreate the dataloader with this sampler
            return DataLoader(
                dataset,
                batch_size=dataloader.batch_size,
                num_workers=dataloader.num_workers,
                pin_memory=dataloader.pin_memory,
                sampler=sampler,
                collate_fn=dataloader.collate_fn,
                drop_last=dataloader.drop_last,
            )
        else:
            return dataloader

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

        # Fit the probe
        self.fit(X_train, y_train)

        # Make predictions
        y_pred = self.predict(X_test)

        # Compute/Log metrics
        self.log_metrics(pl_module, y_pred, y_test)

    def extract_features(self, pl_module, dataloader):
        """Extract features from a dataloader with the BaseEstimator.

        By default, it uses the `transform_step` logic applied on each batch to
        get the embeddings with the labels.
        The input dataloader should yield batches of the form `(X, y)` where X
        is the input data and y is the label.

        Parameters
        ----------
        pl_module: BaseEstimator
            The BaseEstimator module that implements the 'transform_step'.
        dataloader: torch.utils.data.DataLoader
            The dataloader to extract features from. It should yield batches of
            the form `(X, y)` where `X` is the input data and `y` is the label.

        Returns
        -------
        tuple of (z, y)
            Tuple of numpy arrays (z, y) where z are the extracted features
            and y are the corresponding labels.

        """
        is_training = pl_module.training  # Save state

        dataloader = self.adapt_dataloader_for_ddp(
            dataloader, pl_module.trainer
        )

        pl_module.eval()
        X, y = [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                x_batch, y_batch = batch
                x_batch = x_batch.to(pl_module.device)
                y_batch = y_batch.to(pl_module.device)
                features = pl_module.transform_step(
                    x_batch, batch_idx=batch_idx
                )
                X.append(features.detach())
                y.append(y_batch.detach())

        # Concatenate the embeddings
        X = torch.cat(X)
        y = torch.cat(y)

        # Gather across GPUs
        X = pl_module.all_gather(X).cpu().numpy()
        y = pl_module.all_gather(y).cpu().numpy()

        # Reduce (world_size, batch, ...) to (world_size * batch, ...)
        if X.ndim > 2 and pl_module.trainer.world_size > 1:
            X = X.reshape(X.shape[0] * X.shape[1], *X.shape[2:])
            y = y.reshape(y.shape[0] * y.shape[1], *y.shape[2:])

        if is_training:
            pl_module.train()

        return X, y

    def on_train_epoch_end(self, trainer, pl_module):
        self.counter_train_epochs += 1
        if (
            self.every_n_train_epochs is not None
            and self.counter_train_epochs % self.every_n_train_epochs == 0
        ):
            self.probing(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if (
            self.every_n_val_epochs is not None
            and self.counter_val_epochs % self.every_n_val_epochs == 0
        ):
            self.probing(trainer, pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        if self._on_test_epoch_start:
            self.probing(trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if self._on_test_epoch_end:
            self.probing(trainer, pl_module)


class ClassificationProbingCallback(ModelProbing):
    """Perform classification on top of an embedding model.

    Concretely this callback:

    1) Embeds the input data through the torch model.
    2) Fits the classification probe on the embedded data.
    3) Logs the main classification metrics:

       - precision (macro)
       - recall (macro)
       - f1-score (weighted and macro)
       - accuracy (global)
       - balanced accuracy

    Please check this `User Guide <https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report>`_
    for more details on the classification metrics reported.

    Parameters
    ----------
    train_dataloader: torch.utils.data.DataLoader
        Training dataloader yielding batches in the form `(X, y)`
        for further embedding and training of the probe.
    test_dataloader: torch.utils.data.DataLoader
        Test dataloader yielding batches in the form `(X, y)`
        for further embedding and test of the probe.
    probe: sklearn.base.BaseEstimator
        The scikit-learn classifier to be trained on the embedding.
    probe_name: str or None, default=None
        Name of the probe displayed when logging the results.
        It will appear as <probe_name>/<metric_name> for each metric.
        If None,  <probe_class_name>/<metric_name> is displayed.
    every_n_train_epochs: int or None, default=1
        Number of training epochs after which to run the probing.
        Disabled if None.
    every_n_val_epochs: int or None, default=None
        Number of validation epochs after which to run the probing.
        Disabled if None.
    on_test_epoch_start: bool, default=False
        Whether to run the probing at the start of the test epoch.
    on_test_epoch_end: bool, default=False
        Whether to run the probing at the end of the test epoch.
    prog_bar: bool, default=True
        Whether to display the metrics in the progress bar.

    """
    def __init__(
        self,
        train_dataloader,
        test_dataloader,
        probe,
        probe_name=None,
        every_n_train_epochs=1,
        every_n_val_epochs=None,
        on_test_epoch_start=False,
        on_test_epoch_end=False,
        prog_bar=True,
    ):
        if not is_classifier(probe):
            raise ValueError("The probe must be a classifier.")
        super().__init__(
            train_dataloader,
            test_dataloader,
            probe,
            every_n_train_epochs,
            every_n_val_epochs,
            on_test_epoch_start,
            on_test_epoch_end,
            prog_bar,
        )
        self.probe_name = (
            probe_name
            if probe_name is not None
            else f"{probe.__class__.__name__}"
        )

    @rank_zero_only
    def log_metrics(self, pl_module, y_pred, y_true):
        # Compute classification metrics
        metrics_report = classification_report(
            y_true, y_pred, output_dict=True
        )
        # Compute balanced accuracy separately
        bacc = balanced_accuracy_score(y_true, y_pred)

        summary = {
            f"{self.probe_name}/accuracy": metrics_report["accuracy"],
            f"{self.probe_name}/balanced_accuracy": bacc,
            f"{self.probe_name}/f1_macro": metrics_report["macro avg"][
                "f1-score"
            ],
            f"{self.probe_name}/precision_macro": metrics_report["macro avg"][
                "precision"
            ],
            f"{self.probe_name}/recall_macro": metrics_report["macro avg"][
                "recall"
            ],
            f"{self.probe_name}/f1_weighted": metrics_report["weighted avg"][
                "f1-score"
            ],
        }
        pl_module.log_dict(
            summary,
            prog_bar=self.prog_bar,
            on_epoch=True,
        )


class RegressionProbingCallback(ModelProbing):
    """
    Perform regression on top of an embedding model.

    Concretely this callback:

    1) Embeds the input data through the estimator.
    2) Fits the regression probe on the embedded data.
    3) Logs the main regression metrics including:

       - mean absolute error
       - median absolute error
       - root mean squared error
       - mean squared error
       - R² score
       - Pearson's r
       - explained variance score

    Parameters
    ----------
    train_dataloader: torch.utils.data.DataLoader
        Training dataloader yielding batches in the form `(X, y)`
        for further embedding and training of the probe.
    test_dataloader: torch.utils.data.DataLoader
        Test dataloader yielding batches in the form `(X, y)`
        for further embedding and test of the probe.
    probe: sklearn.base.BaseEstimator
        The scikit-learn regressor to be trained on the embedding.
    probe_name: str or None, default=None
        Name of the probe displayed when logging the results.
        It will appear as <probe_name>/<metric_name> for each metric.
        If None,  <probe_class_name>/<metric_name> is displayed.
    every_n_train_epochs: int or None, default=1
        Number of training epochs after which to run the probing.
        Disabled if None.
    every_n_val_epochs: int or None, default=None
        Number of validation epochs after which to run the probing.
        Disabled if None.
    on_test_epoch_start: bool, default=False
        Whether to run the probing at the start of the test epoch.
    on_test_epoch_end: bool, default=False
        Whether to run the probing at the end of the test epoch.
    prog_bar: bool, default=True
        Whether to display the metrics in the progress bar.

    """
    def __init__(
        self,
        train_dataloader,
        test_dataloader,
        probe,
        probe_name=None,
        every_n_train_epochs=1,
        every_n_val_epochs=None,
        on_test_epoch_start=False,
        on_test_epoch_end=False,
        prog_bar=True,
    ):
        if not is_regressor(probe):
            raise ValueError("The probe must be a regressor.")
        super().__init__(
            train_dataloader,
            test_dataloader,
            probe,
            every_n_train_epochs,
            every_n_val_epochs,
            on_test_epoch_start,
            on_test_epoch_end,
            prog_bar,
        )
        self.probe_name = (
            probe_name + "/"
            if probe_name is not None
            else f"{probe.__class__.__name__}/"
        )

    @rank_zero_only
    def log_metrics(self, pl_module, y_pred, y_true):
        # Compute regression metrics
        metrics_report = regression_report(y_true, y_pred, output_dict=True)

        # Log the results
        for name, value in metrics_report.items():
            if isinstance(value, dict):
                value_ = {
                    f"{self.probe_name}{name}/{k}": v
                    for (k, v) in value.items()
                }
                pl_module.log_dict(
                    value_,
                    prog_bar=self.prog_bar,
                    on_epoch=True,
                )
            else:
                pl_module.log(
                    f"{self.probe_name}{name}",
                    value,
                    prog_bar=self.prog_bar,
                    on_epoch=True,
                )
