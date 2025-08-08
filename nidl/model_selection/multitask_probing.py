##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from collections.abc import Iterable
from numbers import Integral
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.utils.data as data
from sklearn.base import BaseEstimator as sk_BaseEstimator
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv, cross_validate
from sklearn.utils.validation import check_array

from nidl.estimators import BaseEstimator, TransformerMixin
from nidl.utils.validation import check_is_fitted


class MultiTaskProbingCV(BaseEstimator):
    """Probe the representation of an embedding estimator on a multi-task
    dataset.

    MultiTaskProbingCV aims at evaluating the representation of an embedding
    model on a dataset with several independent tasks. Input data are embedded
    using a fitted embedding estimator and this representation is evaluated on
    a set of downstream tasks (classification or regression) with one probe
    per task, corresponding to a scikit-learn compatible classifier or
    regressor. A cross-validation scheme is used for evaluating the probe on
    data seen during `fit`.

    MultiTaskProbing implements `fit`, `predict` and `score` to respectively
    fit all probes on the downstream tasks, predict the labels on new data
    and score the probes on new data.

    Parameters
    ----------
    estimator: nidl.estimators.BaseEstimator
        Fitted estimator to evaluate. Must implement `transform`, i.e. inherit
        from :class:`TransformerMixin`.

    tasks: {"classification", "regression"} or list of str
        If "classification" or "regression", all tasks are assumed to be of
        the same type. If a list of str in {"classification", "regression"},
        it defines the list of individual task.

    task_names: str or list of str or None, default=None
        List of unique task names that will be reported in the `cv_results_`.
        It should have the same length as `tasks` (if list).
        If None, ["task1", "task2", ...] are used.

    cv: int, cross-validation generator or iterable, default=5
        Cross-validation splitting strategy for training and testing the
        probes. The training data are splitted in several (train, test) splits
        and the probes are evaluated on these splits. Probes are also  fitted
        on all data for further predictions on new data when called with
        `predict`.
        Possible values are:

        - int to specify the number of folds in a :class:`KFold`.
        - scikit-learn CV splitter.
        - an iterable yielding (train, test) splits as arrays of indices.

        Refer to the `User Guide
        <https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation>`_
        in scikit-learn for the available cross-validation strategies.

    classifier: sklearn.base.BaseEstimator or None, default=None
        The classification probe used to evaluate the data embedding on
        classification tasks. Each probe is trained independently for each
        task. If None (default), logistic regression is used.

    regressor: sklearn.base.BaseEstimator or None, default=None
        The regression probe used to evaluate the data embedding on regression
        tasks. Each probe is trained independently for each task.
        If None (default), ridge regression is used.

    classification_scoring: str, callable, list of str, dict or None,\
        default=None
        Scoring metrics used to evaluate the classification probe across
        cross-validation splits and on new test data.

        If `classification_scoring` represents a single score, one can use:

        - a single string (see `String name scorers
          <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names>`_);
        - a `callable <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-callable>`_
          returning a single value;
        - None, the default probe's evaluation criteria.

        If `classification_scoring` represents multiple scores, one can use:

        - a list of unique strings;
        - a callable returning a dictionary where keys are metric names and
          values are metric scores;
        - a dictionary with metric names as keys and metric scores as values.

    regression_scoring: str, callable, list of str, dict or None, default=None
        Scoring metrics used to evaluate the regression probe across
        cross-validation splits and on new test data. It follows the same
        conventions as `classification_scoring` (for single score and
        multi-scores).

    n_jobs: int or None, default=None
        Number of jobs to run in parallel.  Training the probes and computing
        the score are parallelized over the cross-validation splits.
        `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
        `-1` means using all processors.

    allow_nan: bool, default=False
        If True, NaN values are accepted in the training labels. In that case,
        data with NaN labels are removed during `fit`, which impacts the
        cross-validation splits. If `cv` is an iterable yielding (train, test)
        indices, they are defined over the filtered data. Indices returned by
        `cv_results_` are defined over the original data.

    kwargs: dict
        Additional keyword arguments passed to the parent class
        :class:`nidl.estimators.BaseEstimator`.

    Attributes
    ----------
    ``cv_results_``: dict
        Dictionary storing detailed cross-validation results for each task.
        The structure is as follows:

        {
            <task_name_1>: {
                "test_score": list of float
                    The evaluation score of the probe on each cross-validation
                    test split. Suffix `_score` in `test_score` changes to
                    specific metric name like `test_r2` or `test_accuracy` when
                    multiple scoring metrics are given either in
                    `classification_scoring` or `regression_scoring`.

                "indices": dict
                    A dictionary containing the train/test indices for each CV
                    split:
                    {
                        "train": tuple of arrays
                            Indices of training samples for each split.
                        "test": tuple of arrays
                            Indices of test samples for each split.
                    }

                "fit_time": list of float
                    Time (in seconds) taken to fit the probe on each training
                    split.

                "score_time": list of float
                    Time (in seconds) taken to score the probe on each test
                    split.

                "estimator": list of sklearn.base.BaseEstimator
                    List of probe estimators fitted on each CV training split.
            },

            <task_name_2>: {
                ...
            },
            ...
        }

    ``probe_estimators_``: dict
        Dictionary containing task name as key and the probe (classifier
        or regressor) fitted on the training set as value. These probes
        are used to predict the labels on new data with `predict` method.

    ``scorers_``: dict
        Dictionary containing task name as key and score function used
        to score the probe as value.

    ``n_tasks_``: int
        Number of tasks seen during fit.

    ``n_splits_``: dict
        Dictionary containing task name as key and the number of
        cross-validation splits as value.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        tasks: Union[str, list[str]],
        task_names: Union[list[str], None] = None,
        cv: Union[int, Iterable, object] = 5,
        classifier: Union[sk_BaseEstimator, None] = None,
        regressor: Union[sk_BaseEstimator, None] = None,
        classification_scoring: Optional[
            Union[str, Callable, list[str], dict[str, Callable]]
        ] = None,
        regression_scoring: Optional[
            Union[str, Callable, list[str], dict[str, Callable]]
        ] = None,
        n_jobs: Union[int, None] = None,
        allow_nan: bool = False,
        **kwargs,
    ):
        super().__init__(ignore=["estimator"], **kwargs)
        self.estimator = estimator
        self.tasks = self._parse_tasks(tasks)
        self.task_names = self._parse_task_names(task_names, self.tasks)
        self.cv = self._parse_cv(cv)
        self.classifier = self._parse_classification_probe(classifier)
        self.regressor = self._parse_regression_probe(regressor)
        self.classification_scoring = classification_scoring
        self.regression_scoring = regression_scoring
        self.n_jobs = n_jobs
        self.allow_nan = allow_nan

        self.features_extractor = LabelCachingTransformerWrapper(
            self.estimator
        )

    def fit(
        self,
        train_dataloader: data.DataLoader,
        val_dataloader: Optional[data.DataLoader] = None,
    ):
        """Fit the probes on the training data embedding.

        Parameters
        ----------
        train_dataloader: torch.utils.data.DataLoader
            Training dataloader yielding batches in the form `(X, y)`
            used for further embedding and training of the probes:

            - `X` will be given to `transform_step` of the embedding model.
            - `y` must have shape `(n_samples,)` for singe-task dataset or
              shape `(n_samples, n_targets)` for multi-task dataset. In that
              case, `n_targets` should be equal to the number of tasks.

        val_dataloader: torch.utils.data.DataLoader or None, default=None
            Ignored.

        Returns
        ----------
        self: object
            The fitted estimator.

        """
        X, y = self.extract_features(train_dataloader)
        X = check_array(
            X, ensure_2d=True, dtype="numeric", force_all_finite=True
        )
        y = self._check_y(y, force_all_finite=(not self.allow_nan))

        n_tasks = y.shape[1]

        # Iterate over each task: should it be in parallel?
        self.cv_results_ = {}
        self.probe_estimators_ = {}
        self.scorers_ = {}
        self.n_splits_ = {}
        self.n_tasks_ = n_tasks
        tasks = self._get_tasks(n_tasks)
        task_names = self._get_task_names(n_tasks)

        for i, (task, task_name) in enumerate(zip(tasks, task_names)):
            if task == "classification":
                estimator = self.classifier
                scoring = self.classification_scoring
            elif task == "regression":
                estimator = self.regressor
                scoring = self.regression_scoring

            cv = self.cv
            n_jobs = self.n_jobs
            X_task = X  # do not copy
            y_task = y[:, i]

            self.scorers_[task_name] = check_scoring(
                estimator=estimator, scoring=scoring
            )

            if self.allow_nan:
                y_task, valid_mask, indices = self._filter_nan_or_inf(y_task)
                X_task = X_task[valid_mask]

            self.n_splits_[task_name] = self.cv.get_n_splits(X_task, y_task)

            self.cv_results_[task_name] = cross_validate(
                estimator=estimator,
                X=X_task,
                y=y_task,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
                return_indices=True,
                return_estimator=True,
            )

            if self.allow_nan:  # remap indices to the unfiltered data
                for split in ["train", "test"]:
                    relative_indices = self.cv_results_[task_name]["indices"][
                        split
                    ]
                    absolute_indices = tuple(
                        indices[rel_idx] for rel_idx in relative_indices
                    )
                    self.cv_results_[task_name]["indices"][split] = (
                        absolute_indices
                    )

            # Refit the estimator on the entire dataset
            self.probe_estimators_[task_name] = clone(estimator).fit(
                X_task, y_task
            )
        self.fitted_ = True
        return self

    def predict(self, test_dataloader: data.DataLoader):
        """Predict the labels on the test dataset.

        Parameters
        ----------
        test_dataloader: torch.utils.data.DataLoader
            Testing dataloader yielding batches in the form `(X, y)`.
            `y` is ignored here.

        Returns
        ----------
        y_pred: torch.Tensor, shape `(n_samples, n_targets)`
            The predicted labels across tasks.

        """
        check_is_fitted(self)

        X, _ = self.extract_features(test_dataloader)
        X = check_array(
            X, ensure_2d=True, dtype="numeric", force_all_finite=True
        )
        task_names = self._get_task_names(self.n_tasks_)
        y_pred = [
            self.probe_estimators_[task_name].predict(X)
            for task_name in task_names
        ]

        y_pred = torch.as_tensor(
            check_array(y_pred, ensure_2d=True, force_all_finite=True).T
        )
        return y_pred

    def score(self, test_dataloader: data.DataLoader):
        """Score the probes on the test dataset.

        Parameters
        ----------
        test_dataloader: torch.utils.data.DataLoader
            Testing dataloader yielding batches in the form `(X, y)`.
            `y` must have shape `(n_samples, n_targets)` for multi-task dataset
            or `(n_samples,)` for single-task dataset.

        Returns
        ----------
        scores: dict
            Scores over the test data with task name as key and score as value.
            If multiple scoring metrics are provided for a task (classification
            or regression), a dictionary with the metric name as key and score
            as value is returned for the given task.

        """
        check_is_fitted(self)
        X, y = self.extract_features(test_dataloader)
        X = check_array(
            X, ensure_2d=True, dtype="numeric", force_all_finite=True
        )
        y = self._check_y(y, force_all_finite=True)

        n_tasks = y.shape[1]
        if n_tasks != self.n_tasks_:
            raise ValueError(
                f"Expected {self.n_tasks_} tasks but got {n_tasks}."
            )

        task_names = self._get_task_names(n_tasks)
        scores = {}
        for i, task_name in enumerate(task_names):
            scorer = self.scorers_[task_name]
            estimator = self.probe_estimators_[task_name]
            scores[task_name] = scorer(estimator, X, y[:, i])
        return scores

    def extract_features(self, dataloader: data.DataLoader):
        """Extract features and labels using the embedding estimator.

        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            The dataloader to extract features from. It should yield batches of
            the form `(X, y)` where X is the input data and y are the labels.

        Returns
        ----------
        tuple of (array, array)
            Tuple of numpy arrays `(X, y)` where `X` are the extracted features
            and `y` are the corresponding labels.

        """

        X = self.features_extractor.transform(dataloader)
        y = self.features_extractor.get_labels()
        X = X.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        return X, y

    def _get_tasks(self, n_tasks) -> list[str]:
        tasks = self.tasks

        if isinstance(tasks, str):
            tasks = n_tasks * [tasks]

        if len(tasks) != n_tasks:
            raise ValueError(
                f"Expected {n_tasks} tasks but `tasks` has {len(tasks)} "
                "elements."
            )
        return tasks

    def _get_task_names(self, n_tasks) -> list[str]:
        task_names = self.task_names

        if task_names is None:
            return [f"task{i}" for i in range(n_tasks)]

        if len(task_names) != n_tasks:
            raise ValueError(
                f"Expected {n_tasks} task names but `task_names` has "
                f"{len(task_names)} elements."
            )
        return task_names

    @staticmethod
    def _filter_nan_or_inf(y):
        """Filter out rows in y containing NaN or infinite values.

        Parameters
        ----------
        y: ndarray with shape (n_samples, *)
            Input labels eventually containing NaN or infinite values.

        Returns
        ----------
        y_filtered : ndarray of shape `(n_samples', *)`
            y with invalid rows removed.
        valid_mask : boolean array of shape `(n_samples,)`
            Boolean mask of valid rows.
        indices: array of shape `(n_samples',)`
            Indices of valid rows in the original y.
        """
        # Handle 1d separately for clarity
        if y.ndim == 1:
            valid_mask = np.isfinite(y)
        else:
            valid_mask = np.all(np.isfinite(y), axis=1)

        indices = np.where(valid_mask)[0]
        return y[valid_mask], valid_mask, indices

    @staticmethod
    def _check_y(y, force_all_finite: bool = True):
        """Check that y is a 1d or 2d array and cast it to 2d array."""
        y = check_array(y, ensure_2d=False, force_all_finite=force_all_finite)
        if y.ndim == 1:
            y = y.reshape(len(y), -1)

        if y.ndim != 2:
            raise ValueError(
                f"Expected 1d or 2d array for `y`, got {y.ndim}d."
            )

        return y

    @staticmethod
    def _parse_tasks(tasks):
        allowed = {"classification", "regression"}
        if isinstance(tasks, str):
            if tasks not in allowed:
                raise ValueError(
                    f"If string, `tasks` must be one of {allowed}."
                )
            return tasks
        elif isinstance(tasks, (list, tuple)):
            if not all(t in allowed for t in tasks):
                raise ValueError(
                    f"All elements in `tasks` must be one of {allowed}."
                )
            return list(tasks)
        else:
            raise TypeError("`tasks` must be a string or a list of strings.")

    @staticmethod
    def _parse_task_names(task_names, tasks):
        if task_names is None:
            return None  # Cannot determine the number of tasks here
        if isinstance(task_names, str):
            task_names = [task_names]
        if not isinstance(task_names, list) or not all(
            isinstance(tn, str) for tn in task_names
        ):
            raise TypeError("`task_names` must be a list of strings or None.")
        if isinstance(tasks, list) and len(task_names) != len(tasks):
            raise ValueError(
                "Length of `task_names` must match length of `tasks`."
            )
        if len(set(task_names)) != len(task_names):
            raise ValueError("All `task_names` must be unique.")
        return task_names

    @staticmethod
    def _parse_cv(cv):
        if (
            isinstance(cv, Integral)
            or hasattr(cv, "split")
            or hasattr(cv, "__iter__")
        ):
            return check_cv(cv)
        else:
            raise TypeError(
                "`cv` must be an int, a cross-validation splitter, or an "
                "iterable of (train, test) splits."
            )

    @staticmethod
    def _parse_classification_probe(probe):
        if probe is None:
            return LogisticRegression()
        if not is_classifier(probe) or not isinstance(probe, sk_BaseEstimator):
            raise TypeError(
                "`classification_probe` must be a scikit-learn classifier or "
                "None."
            )
        return probe

    @staticmethod
    def _parse_regression_probe(probe):
        if probe is None:
            return Ridge()
        if not is_regressor(probe) or not isinstance(probe, sk_BaseEstimator):
            raise TypeError(
                "`regression_probe` must be a scikit-learn regressor or None."
            )
        return probe


class LabelCachingTransformerWrapper(TransformerMixin, BaseEstimator):
    """A wrapper for transformer-based estimators that caches labels during
    transform calls.

    This class wraps an existing estimator implementing `transform_step`
    (e.g., an embedding estimator) so that labels from each processed
    batch are automatically stored during `.transform()` calls. This is
    useful when working with dataloaders that yield `(X, y)` pairs and
    you need both the transformed features and their corresponding labels
    after the transformation.


    Parameters
    ----------
    estimator: nidl.estimators.BaseEstimator
        The BaseEstimator module that implements the `transform_step`.

    kwargs: dict
        Ignored. This is only to match the signature of BaseEstimator.
    """

    def __init__(self, estimator: BaseEstimator, **kwargs):
        trainer_params_ = {
            **estimator.trainer_params_,
            "callbacks": None,
            "enable_checkpointing": False,
            "enable_progress_bar": True,
            "logger": False,
            "ignore": ["estimator"],
        }
        super().__init__(**trainer_params_)

        if not isinstance(estimator, TransformerMixin):
            raise TypeError(
                "`estimator` must be a subclass of TransformerMixin."
            )

        self.estimator = estimator
        self.fitted_ = estimator.fitted_
        self._cached_labels = []

    def transform(self, test_dataloader: data.DataLoader):
        # clear the cached labels before each transform
        self._cached_labels.clear()
        self.fitted_ = self.estimator.fitted_
        return super().transform(test_dataloader)

    def transform_step(
        self,
        batch: tuple[torch.Tensor],
        batch_idx: int = 0,
        dataloader_idx: int = 0,
    ):
        x_batch, y_batch = batch
        self._cached_labels.append(y_batch.detach())
        x_features = self.estimator.transform_step(
            x_batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
        )
        return x_features

    def get_labels(self):
        """Get the cached labels."""
        return torch.cat(self._cached_labels, dim=0)
