##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from typing import Union

import numpy as np
from pytorch_lightning.utilities import rank_zero_only
from sklearn.base import BaseEstimator as sk_BaseEstimator
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.metrics import balanced_accuracy_score, classification_report
from torch.utils.data import DataLoader

from nidl.callbacks.model_probing import ModelProbing
from nidl.metrics import regression_report


class MultiTaskEstimator(sk_BaseEstimator):
    """
    A meta-estimator that wraps a list of sklearn estimators
    for multi-task problems (mixed regression/classification).

    Parameters
    ----------
    estimators: list of sklearn.base.BaseEstimator
        List of scikit-learn estimators (classifiers, regressors or a mixed).
        It sets the number of tasks to be solved.

    """

    def __init__(self, estimators):
        for est in estimators:
            if not (is_classifier(est) or is_regressor(est)):
                raise ValueError(
                    "All estimators must be classifier or regressor, got "
                    f"{est}"
                )
        self.estimators = estimators

    def fit(self, X, y):
        """
        Fit each estimator on its corresponding column in y.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples, n_tasks)

        Returns
        ----------
        self: the fitted estimator.
        """
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if y.ndim != 2:
            raise ValueError(f"'y' must be 2d, got {y.ndim}d")

        n_tasks = len(self.estimators)
        if y.shape[1] != n_tasks:
            raise ValueError(
                f"'y' must have {n_tasks} columns, got {y.shape[0]}"
            )

        self.estimators_ = []
        for i, est in enumerate(self.estimators):
            fitted = clone(est).fit(X, y[:, i])
            self.estimators_.append(fitted)
        return self

    def predict(self, X):
        """
        Predict for each task, returns array of shape (n_samples, n_tasks).

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)

        Returns
        ----------
        y_preds: array-like, shape (n_samples, n_tasks)

        """
        preds = [est.predict(X).reshape(-1, 1) for est in self.estimators_]
        return np.hstack(preds)

    def score(self, X, y):
        """
        Average score across all tasks.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Test samples.

        y: array-like, shape (n_samples, n_tasks)
            True targets for X.

        Returns
        ----------
        score: float
        """
        y = np.asarray(y)
        scores = []
        for i, est in enumerate(self.estimators_):
            scores.append(est.score(X, y[:, i]))
        return np.mean(scores)

    def __len__(self):
        return len(self.estimators)


class MultitaskModelProbing(ModelProbing):
    """Callback to probe the representation of an embedding estimator on a
    multi-task dataset.

    This callback implements multitask probing on top of an embedding estimator
    for both classification and regression tasks. It avoids computing the
    embeddings multiple times for each task by storing the embeddings once in
    memory. Each probe is then trained and evaluated separately on the stored
    embeddings for each task.

    Parameters
    ----------
    train_dataloader: torch.utils.data.DataLoader
        Training dataloader yielding batches in the form `(X, y)`
        for further embedding and training of the probe.
        `y` should have shape `(n_samples, n_tasks)` with one output per task
        (categorical or continuous).

    test_dataloader: torch.utils.data.DataLoader
        Test dataloader yielding batches in the form `(X, y)`
        for further embedding and test of the probe. `y` should have shape
        `(n_samples, n_tasks)` with one output per task.

    probes: list of sklearn.base.BaseEstimator or MultiTaskEstimator
        The probes used to evaluate the data embedding on multiple tasks
        (classification or regression). Each probe is fitted on one task
        (=one target) and should implement `fit` and `predict`.

    probe_names: str or list of str or None, default=None
        Name of the probes to be displayed when logging the results.
        It will appear as <task_name_i>/<metric_name> for each task `i`.
        It should have the same length as `probes` (if list).
        If None, ["task1", "task2", ...] are used.

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
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        probes: Union[list[sk_BaseEstimator], MultiTaskEstimator],
        probe_names: Union[list[str], None] = None,
        every_n_train_epochs: Union[int, None] = 1,
        every_n_val_epochs: Union[int, None] = None,
        on_test_epoch_start: bool = False,
        on_test_epoch_end: bool = False,
        prog_bar: bool = True,
    ):
        if isinstance(probes, list):
            probes = MultiTaskEstimator(probes)

        if not isinstance(probes, MultiTaskEstimator):
            raise ValueError(
                "'probes' must be a list of scikit-learn estimators "
                "(classifier or regressor) or a 'MultiTaskEstimator, "
                f"got {type(probes)}"
            )

        self.probe_names = self._parse_names(probes, probe_names)

        super().__init__(
            train_dataloader,
            test_dataloader,
            probes,
            every_n_train_epochs,
            every_n_val_epochs,
            on_test_epoch_start,
            on_test_epoch_end,
            prog_bar,
        )

    def log_classification_metrics(self, pl_module, y_pred, y_true, task_name):
        """Log the metrics for a classification task.

        The main classification metrics reported are:

        - precision (macro)
        - recall (macro)
        - f1-score (weighted and macro)
        - accuracy (global)
        - balanced accuracy

        Parameters
        ----------
        pl_module: nidl.estimators.base.BaseEstimator
            The embedding estimator currently evaluated.

        y_pred: array-like, shape (n_samples,)
            Predicted values for classification.

        y_true: array-like, shape (n_samples,)
            Ground-truth values.

        task_name: str
            Name to display when logging the metrics.
        """
        # Compute classification metrics
        metrics_report = classification_report(
            y_true, y_pred, output_dict=True
        )
        # Compute balanced accuracy separately
        bacc = balanced_accuracy_score(y_true, y_pred)

        summary = {
            f"{task_name}/accuracy": metrics_report["accuracy"],
            f"{task_name}/balanced_accuracy": bacc,
            f"{task_name}/f1_macro": metrics_report["macro avg"]["f1-score"],
            f"{task_name}/precision_macro": metrics_report["macro avg"][
                "precision"
            ],
            f"{task_name}/recall_macro": metrics_report["macro avg"]["recall"],
            f"{task_name}/f1_weighted": metrics_report["weighted avg"][
                "f1-score"
            ],
        }
        pl_module.log_dict(
            summary,
            prog_bar=self.prog_bar,
            on_epoch=True,
        )

    def log_regression_metrics(self, pl_module, y_pred, y_true, task_name):
        """Log the metrics for a regression task.

        The main regression metrics reported are:

        - mean absolute error
        - median absolute error
        - root mean squared error
        - mean squared error
        - R² score
        - Pearson's r
        - explained variance score

        Parameters
        ----------
        pl_module: nidl.estimators.base.BaseEstimator
            The embedding estimator currently evaluated.

        y_pred: array-like, shape (n_samples,)
            Predicted values for regression.

        y_true: array-like, shape (n_samples,)
            Ground-truth values.

        task_name: str
            Name to display when logging the metrics.
        """
        # Compute regression metrics
        metrics_report = regression_report(y_true, y_pred, output_dict=True)

        # Log the results
        for name, value in metrics_report.items():
            if isinstance(value, dict):
                value_ = {
                    f"{task_name}/{name}/{k}": v for (k, v) in value.items()
                }
                pl_module.log_dict(
                    value_,
                    prog_bar=self.prog_bar,
                    on_epoch=True,
                )
            else:
                pl_module.log(
                    f"{task_name}/{name}",
                    value,
                    prog_bar=self.prog_bar,
                    on_epoch=True,
                )

    @rank_zero_only
    def log_metrics(self, pl_module, y_pred, y_true):
        """Log the metrics for each task (classification or regression).

        Parameters
        ----------
        pl_module: nidl.estimators.base.BaseEstimator
            The embedding estimator currently evaluated.

        y_pred: array-like, shape (n_samples, n_tasks)
            Predicted values.

        y_true: array-like, shape (n_samples, n_tasks)
            Ground-truth for the tasks.

        """
        for i, task_name in enumerate(self.probe_names):
            if is_classifier(self.probe.estimators[i]):
                self.log_classification_metrics(
                    pl_module, y_pred[:, i], y_true[:, i], task_name
                )
            else:
                self.log_regression_metrics(
                    pl_module, y_pred[:, i], y_true[:, i], task_name
                )

    def _parse_names(self, probes, names):
        if names is None:
            return [f"task{i}" for i in range(len(probes))]
        elif isinstance(names, list):
            if len(names) != len(probes):
                raise ValueError(
                    "Invalid number of probe names: "
                    f"{len(probes)} != {len(names)})"
                )
            return names
        raise ValueError(f"Invalid type of probes names: {type(names)}")
