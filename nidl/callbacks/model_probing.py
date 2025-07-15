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
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array
from torch.utils.data import DataLoader

from nidl.estimators.base import BaseEstimator
from nidl.metrics import regression_report
from nidl.utils.validation import _estimator_is


class ModelProbing(ABC, pl.Callback):
    """Callback implementing the basic logic of embedding model's probing.

    1) Embeds the input data (training+test) through the estimator using
       `transform_step` method.
    2) Train the probe on the training embedding
    3) Test the probe on the test embedding and log the metrics

    This callback is abstract and should be inherited to implement
    the `fit`, `predict` and `log_metrics` methods (e.g. for Ridge regression,
    KNN regression, logistic regression, KNN classification, ...).

    Parameters
    ----------
    train_dataloader: torch.utils.data.DataLoader
        Training dataloader yielding batches in the form (X, y)
        for further embedding and training of the probe.

    test_dataloader: torch.utils.data.DataLoader
        Test dataloader yielding batches in the form (X, y)
        for further embedding and test of the probe.

    probe_name: str or None, default=None
        Name of the probe displayed when logging the results.
        It will appear as <probe_name>/<metric_name> for each metric.
        If None, only <metric_name> is displayed.

    every_n_train_epochs: int or None, default=1
        Number of training epochs after which to run the linear probing.
        Disabled if None.

    every_n_val_epochs: int or None, default=None
        Number of validation epochs after which to run the linear probing.
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
        probe_name: Union[str, None] = None,
        every_n_train_epochs: Union[int, None] = 1,
        every_n_val_epochs: Union[int, None] = None,
        on_test_epoch_start: bool = False,
        on_test_epoch_end: bool = False,
        prog_bar: bool = True,
    ):
        super().__init__()
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.probe_name = probe_name + "/" if probe_name is not None else ""
        self.every_n_train_epochs = every_n_train_epochs
        self.every_n_val_epochs = every_n_val_epochs
        self._on_test_epoch_start = on_test_epoch_start
        self._on_test_epoch_end = on_test_epoch_end
        self.prog_bar = prog_bar
        self.counter_train_epochs = 0
        self.counter_val_epochs = 0

    @abstractmethod
    def fit(self, X, y):
        """Fit the probe on the embeddings and labels of the training data."""

    @abstractmethod
    def predict(self, X):
        """Predict the probe on new data X."""

    @abstractmethod
    def log_metrics(self, pl_module, y_pred, y_true):
        """Log the metrics given the predictions and the true labels."""

    def linear_probing(self, pl_module: BaseEstimator):
        """Perform the linear probing on the given estimator.

        This method performs the following steps:
        1) Extracts the features from the training and test dataloaders
        2) Fits the probe on the training features and labels
        3) Makes predictions on the test features
        4) Computes and logs the metrics.

        Parameters
        ----------
        pl_module: BaseEstimator
            The BaseEstimator module that implements the 'transform_step'.

        Raises
        ------
        ValueError: If the pl_module does not inherit from BaseEstimator or
        from TransformerMixin.

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
        The input dataloader should yield batches of the form (X, y) where X
        is the input data and y is the label.

        Parameters
        ----------
        pl_module: BaseEstimator
            The BaseEstimator module that implements the 'transform_step'.

        dataloader: torch.utils.data.DataLoader
            The dataloader to extract features from. It should yield batches of
            the form (X, y) where X is the input data and y is the label.

        Returns
        -------
        tuple of (X, y)
            Tuple of numpy arrays (X, y) where X is the extracted features
            and y is the corresponding labels.

        """
        is_training = pl_module.training  # Save state

        pl_module.eval()
        X, y = [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                x_batch, y_batch = batch
                x_batch = x_batch.to(pl_module.device)
                features = pl_module.transform_step(
                    x_batch, batch_idx=batch_idx
                )
                X.append(features.detach().cpu())
                y.append(y_batch.detach().cpu())
        X = torch.cat(X).numpy()
        y = torch.cat(y).numpy()

        if is_training:
            pl_module.train()

        return X, y

    def on_train_epoch_end(self, trainer, pl_module):
        self.counter_train_epochs += 1
        if (
            self.every_n_train_epochs is not None
            and self.counter_train_epochs % self.every_n_train_epochs == 0
        ):
            self.linear_probing(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if (
            self.every_n_val_epochs is not None
            and self.counter_val_epochs % self.every_n_val_epochs == 0
        ):
            self.linear_probing(pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        if self._on_test_epoch_start:
            self.linear_probing(pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if self._on_test_epoch_end:
            self.linear_probing(pl_module)


class RidgeCVCallback(ModelProbing):
    """Perform Ridge regression on top of an embedding model.

    Concretely this callback:

    1) Embeds the input data through the estimator.

    2) Performs n-fold CV to find the best L2 regularization strength.

    3) Logs the main regression metrics by regressor and averaged,
       including:

       - mean absolute error
       - median absolute error
       - root mean squared error
       - mean squared error
       - r2 score
       - pearsonr
       - explained variance score

       If multiple regressors are given (multivariate regression),
       metrics are computed per regressor and averaged.

    Parameters
    ----------
    train_dataloader: torch.utils.data.DataLoader
        Training dataloader yielding batches in the form (X, y)
        for further embedding and training of the ridge probe.

    test_dataloader: torch.utils.data.DataLoader
        Test dataloader yielding batches in the form (X, y)
        for further embedding and test of the ridge probe.

    probe_name: str or None, default=None
        Name of the probe displayed when logging the results.
        It will appear as <probe_name>/<metric_name> for each metric.
        If None, only <metric_name> is displayed.

    alphas : tuple of floats, default=(0.1, 1.0, 10.0)
        Arrays of `alpha` values to try in CV. It corresponds to the
        regularization strength.

    cv: int or cross-validation generator, default=5
        How many folds to use for cross-validating the `alpha`
        regularization strength in the `Ridge` regression.

    scoring: str in {"r2", "neg_mean_absolute_error",
        "neg_mean_squared_error", ...}, default="r2"
        Which scoring function to use to cross-validate the `alpha`
        hyper-parameter. For a complete list of scoring options, check
        https://scikit-learn.org/1.4/modules/model_evaluation.html#scoring

    kwargs: dict
        Additional keyword arguments to pass to the `ModelProbing` constructor
        (e.g. `every_n_train_epochs`, `every_n_val_epochs`, `prog_bar`, ...).

    """

    def __init__(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        probe_name: Union[str, None] = None,
        alphas: tuple[float] = (0.1, 1.0, 10.0),
        cv: int = 5,
        scoring: str = "r2",
        **kwargs,
    ):
        super().__init__(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            probe_name=probe_name,
            **kwargs,
        )
        self.alphas = alphas
        self.cv = cv
        self.scoring = scoring
        self.probe = RidgeCV(
            alphas=self.alphas,
            cv=self.cv,
            scoring=self.scoring,
        )

    def fit(self, X, y):
        return self.probe.fit(X, y)

    def predict(self, X):
        return self.probe.predict(X)

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


class KNeighborsRegressorCVCallback(ModelProbing):
    """Perform KNN regression on top of an embedding model.

    Concretely this callback:
      1) Embeds the input data through the torch model
      2) Performs n-fold CV to find the best `n_neighbors` neighbors
      3) Log the main regression metrics by regressor and averaged, including:
            * mean absolute error
            * median absolute error
            * root mean squared error
            * mean squared error
            * r2 score
            * pearsonr
            * explained variance score

    Parameters
    ----------
    train_dataloader: torch.utils.data.DataLoader
        Training dataloader yielding batches in the form (X, y)
        for further embedding and training of the KNN probe.

    test_dataloader: torch.utils.data.DataLoader
        Test dataloader yielding batches in the form (X, y)
        for further embedding and test of the KNN probe.

    probe_name: str or None, default=None
        Name of the probe displayed when logging the results.
        It will appear as <probe_name>/<metric_name> for each metric.
        If None, only <metric_name> is displayed.

    n_neighbors: tuple of int, default=(2, 5, 10)
        Arrays of `n_neighbors` values to try in CV.
        It corresponds to the number of neighbors to use by the KNN on the
        training set.

    cv: int or cross-validation generator, default=5
        How many folds to use for cross-validating the `alpha`
        regularization strength in the `Ridge` regression.

    n_jobs: int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context.
        ``-1`` means using all processors.

    scoring: str in {"r2", "neg_mean_absolute_error",
        "neg_mean_squared_error",  ...}, default="r2"
        Which scoring function to use to cross-validate the `n_neighbors`
        hyper-parameter. For a complete list of scoring options, check
        https://scikit-learn.org/1.4/modules/model_evaluation.html#scoring

    kwargs: dict
        Additional keyword arguments to pass to the `ModelProbing` constructor
        (e.g. `every_n_train_epochs`, `every_n_val_epochs`, `prog_bar`, ...).

    """

    def __init__(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        probe_name: Union[str, None] = None,
        n_neighbors: tuple[int] = (2, 5, 10),
        cv: int = 5,
        n_jobs: Union[int, None] = None,
        scoring: str = "r2",
        **kwargs,
    ):
        super().__init__(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            probe_name=probe_name,
            **kwargs,
        )
        self.n_neighbors = n_neighbors
        self.cv = cv
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.probe = GridSearchCV(
            KNeighborsRegressor(),
            param_grid={"n_neighbors": self.n_neighbors},
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            cv=self.cv,
        )

    def fit(self, X, y):
        """Fit the k-nearest neighbors regressor from the training dataset and
        log the regression metrics.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : {array-like, sparse matrix} of shape (n_samples,) or \
                (n_samples, n_outputs)
            Target values.

        """
        return self.probe.fit(X, y)

    def predict(self, X):
        return self.probe.predict(X)

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


class LogisticRegressionCVCallback(ModelProbing):
    """Performs logistic regression on top of an embedding model.

    Concretely this callback:

    1) Embeds the input data through the torch model.

    2) Performs n-fold CV to find the best L2 regularization strength.

    3) Logs the main classification metrics for each class and averaged
       across classes (weighted by class support and unweighted):

       - precision
       - recall
       - f1-score
       - support
       - accuracy (global)

    Please check this `User Guide <https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report>`_
    for more details on the classification metrics reported.

    Parameters
    ----------
    train_dataloader: torch.utils.data.DataLoader
        Training dataloader yielding batches in the form (X, y)
        for further embedding and training of the logistic regression
        probe.

    test_dataloader: torch.utils.data.DataLoader
        Test dataloader yielding batches in the form (X, y)
        for further embedding and test of the logistic regression
        probe.

    probe_name: str or None, default=None
        Name of the probe displayed when logging the results.
        It will appear as <probe_name>/<metric_name> for each metric.
        If None, only <metric_name> is displayed.

    Cs : int or list of floats, default=10
        Each of the values in Cs describes the inverse of regularization
        strength. If Cs is as an int, then a grid of Cs values are chosen
        in a logarithmic scale between 1e-4 and 1e4.
        Like in support vector machines, smaller values specify stronger
        regularization.

    cv: int or cross-validation generator, default=5
        How many folds to use for cross-validating the `C`
        regularization strenght
        in the `LogisticRegression`.

    max_iter: int, default=100
        Maximum number of iterations taken for the solver to converge.

    n_jobs: int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context.
        ``-1`` means using all processors.

    scoring: str in {"accuracy", "balanced_accuracy", "f1", ...},
        default="balanced_accuracy"
        Which scoring function to use to cross-validate the `C`
        hyper-parameter.
        For a complete list of scoring options, check
        https://scikit-learn.org/1.4/modules/model_evaluation.html#scoring

    linear_solver: str in {'lbfgs', 'liblinear', 'newton-cg',
        'newton-cholesky', 'sag', 'saga'}, default='lbfgs'
        Algorithm to use in the optimization problem.

    kwargs: dict
        Additional keyword arguments to pass to the `ModelProbing` constructor
        (e.g. `every_n_train_epochs`, `every_n_val_epochs`, `prog_bar`, ...).

    """

    def __init__(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        probe_name: Union[str, None] = None,
        Cs: Union[int, list[float]] = 5,
        cv: int = 5,
        max_iter: int = 100,
        n_jobs: Union[int, None] = None,
        scoring: str = "balanced_accuracy",
        linear_solver: str = "lbfgs",
        **kwargs,
    ):
        super().__init__(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            probe_name=probe_name,
            **kwargs,
        )
        self.Cs = Cs
        self.cv = cv
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.linear_solver = linear_solver
        self.probe = LogisticRegressionCV(
            Cs=self.Cs,
            cv=self.cv,
            penalty="l2",
            solver=self.linear_solver,
            max_iter=self.max_iter,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )

    def fit(self, X, y):
        """Fit the model according to the given training data and log
        the classification metrics.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        """
        return self.probe.fit(X, y)

    def predict(self, X):
        return (self.probe.predict(X), self.probe.predict_proba(X))

    def log_metrics(self, pl_module, y_pred, y_true):
        labels = unique_labels(y_pred[0], y_true)

        # Compute classification metrics
        metrics_report = classification_report(
            y_true,
            y_pred[0],
            output_dict=True,
            target_names=[f"class {i}" for i in range(len(labels))],
        )

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


class KNeighborsClassifierCVCallback(ModelProbing):
    """Performs KNN classification on top of an embedding model.
    Concretely this callback:

    1) Embeds the input data through the torch model.

    2) Performs n-fold CV to find the best `n_neighbors` neighbors.

    3) Logs the main classification metrics for each class and averaged
       across classes (weighted by class support and unweighted):

       - precision
       - recall
       - f1-score
       - support
       - accuracy (global)
        
    Please check this `User Guide <https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report>`_
    for more details on the classification metrics reported.

    Parameters
    ----------
    train_dataloader: torch.utils.data.DataLoader
        Training dataloader yielding batches in the form (X, y)
        for further embedding and training of the KNN probe.

    test_dataloader: torch.utils.data.DataLoader
        Test dataloader yielding batches in the form (X, y)
        for further embedding and test of the KNN probe.
                
    probe_name: str or None, default=None
        Name of the probe displayed when logging the results.
        It will appear as <probe_name>/<metric_name> for each metric.
        If None, only <metric_name> is displayed.

    n_neighbors: tuple of int, default=(2, 5, 10)
        Array of `n_neighbors` values to try in CV.
        It corresponds to the number of neighbors to use by the KNN on
        the training set.

    cv: int or cross-validation generator, default=5
        How many folds to use for cross-validating the `n_neighbors`
        hyper-parameter.

    n_jobs: int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    scoring: str in {"accuracy", "balanced_accuracy", "f1", ...}, \
        default="balanced_accuracy"
        Which scoring function to use to cross-validate the `n_neighbors`
        hyper-parameter.
        For a complete list of scoring options, check
        https://scikit-learn.org/1.4/modules/model_evaluation.html#scoring

    kwargs: dict
        Additional keyword arguments to pass to the `ModelProbing` constructor
        (e.g. `every_n_train_epochs`, `every_n_val_epochs`, `prog_bar`, ...).

    """

    def __init__(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        probe_name: Union[str, None] = None,
        n_neighbors: tuple[int] = (2, 5, 10),
        cv: int = 5,
        n_jobs: Union[int, None] = None,
        scoring: str = "balanced_accuracy",
        **kwargs,
    ):
        super().__init__(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            probe_name=probe_name,
            **kwargs,
        )
        self.n_neighbors = n_neighbors
        self.cv = cv
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.probe = GridSearchCV(
            KNeighborsClassifier(),
            param_grid={"n_neighbors": self.n_neighbors},
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            cv=self.cv,
        )

    def fit(self, X, y):
        """Fit the k-nearest neighbors classifier from the training dataset and
        log the classification metrics.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric='precomputed'
            Training data.

        y : {array-like, sparse matrix} of shape (n_samples,) or \
                (n_samples, n_outputs)
            Target values.

        """
        return self.probe.fit(X, y)

    def predict(self, X):
        return (self.probe.predict(X), self.probe.predict_proba(X))

    def log_metrics(self, pl_module, y_pred, y_true):
        # Compute classification metrics
        labels = unique_labels(y_pred[0], y_true)
        metrics_report = classification_report(
            y_true,
            y_pred[0],
            output_dict=True,
            labels=labels,
            target_names=[f"class {i}" for i in range(len(labels))],
        )

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
