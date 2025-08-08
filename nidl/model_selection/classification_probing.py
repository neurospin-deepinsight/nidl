##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from collections.abc import Iterable
from typing import Callable, Optional, Union

import torch
import torch.utils.data as data
from sklearn.base import BaseEstimator as sk_BaseEstimator
from sklearn.base import clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import cross_validate
from sklearn.utils.validation import check_array

from nidl.estimators import BaseEstimator
from nidl.model_selection.multitask_probing import MultiTaskProbingCV
from nidl.utils.validation import check_is_fitted


class ClassificationProbingCV(MultiTaskProbingCV):
    """Probe the representation of an embedding estimator on a classification
    task.

    ClassificationProbingCV aims at evaluating the representation of an
    embedding model on a classification dataset. Input data are embedded
    using a fitted embedding estimator and this representation is evaluated on
    the classification task using a scikit-learn compatible classifier. A
    cross-validation scheme is used for evaluating the classifier on data
    seen during `fit`.

    ClassificationProbingCV implements `fit`, `predict` and `score` to
    respectively fit the classifier on the task, predict the labels on new
    data and score the classifier on new data.

    Parameters
    ----------
    estimator: nidl.estimators.BaseEstimator
        Fitted estimator to evaluate. Must implement `transform`, i.e. inherit
        from :class:`TransformerMixin`.

    cv: int, cross-validation generator or iterable, default=5
        Cross-validation splitting strategy for training and testing the
        classifier. The training data are splitted in several (train, test)
        splits and the classifier is evaluated on these splits. Classifier is
        also fitted on all data for further predictions on new data when called
        with `predict`.
        Possible values are:

        - int to specify the number of folds in a :class:`KFold`.
        - scikit-learn CV splitter.
        - an iterable yielding (train, test) splits as arrays of indices.

        Refer to the `User Guide
        <https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation>`_
        in scikit-learn for the available cross-validation strategies.

    classifier: sklearn.base.BaseEstimator or None, default=None
        The classification probe used to evaluate the data embedding on the
        classification task. If None (default), a logistic regression is used.

    scoring: str, callable, list of str, dict or None, default=None
        Scoring metrics used to evaluate the classification probe across
        cross-validation splits and on new test data.

        If `scoring` represents a single score, one can use:

        - a single string (see `String name scorers
          <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names>`_);
        - a `callable <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-callable>`_
          returning a single value;
        - None, the default probe's evaluation criteria (accuracy).

        If `scoring` represents multiple scores, one can use:

        - a list of unique strings;
        - a callable returning a dictionary where keys are metric names and
          values are metric scores;
        - a dictionary with metric names as keys and metric scores as values.

    n_jobs: int or None, default=None
        Number of jobs to run in parallel.  Training the classifier and
        computing the score are parallelized over the cross-validation splits.
        `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
        `-1` means using all processors.

    allow_nan: bool, default=False
        If True, NaN values are accepted in the training labels. In that case,
        data with NaN labels are removed during `fit`, which impacts the
        cross-validation splits. If `cv` is an iterable yielding (train, test)
        indices, they are defined over the filtered data. Indices returned by
        `cv_results_` are defined over the original data.

    kwargs: dict
        Additional keyword arguments passed to the parent class.

    Attributes
    ----------
    ``cv_results_``: dict
        Dictionary storing detailed cross-validation results for the
        classification task. The structure is as follows:

        {
            "test_score": list of float
                The evaluation score of the classifier on each cross-validation
                test split. Suffix `_score` in `test_score` changes to
                specific metric name like `test_accuracy` when multiple scoring
                metrics are given either in `scoring`.

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
                Time (in seconds) taken to fit the classifier on each training
                split.

            "score_time": list of float
                Time (in seconds) taken to score the classifier on each test
                split.

            "estimator": sklearn.base.BaseEstimator
                Classifier object fitted on each CV training split.
        }

    ``classifier_``: sklearn.base.BaseEstimator
        Classifier fitted on the training set.

    ``scorer_``: callable
        Score function used to score the classifier.

    ``n_splits_``: int
        Number of cross-validation splits.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        cv: Union[int, Iterable, object] = 5,
        classifier: Union[sk_BaseEstimator, None] = None,
        scoring: Optional[
            Union[str, Callable, list[str], dict[str, Callable]]
        ] = None,
        n_jobs: Union[int, None] = None,
        allow_nan: bool = False,
        **kwargs,
    ):
        super().__init__(
            estimator=estimator,
            tasks="classification",
            task_names=None,
            cv=cv,
            classifier=classifier,
            regressor=None,
            classification_scoring=scoring,
            regression_scoring=None,
            n_jobs=n_jobs,
            allow_nan=allow_nan,
            **kwargs,
        )

    def fit(
        self,
        train_dataloader: data.DataLoader,
        val_dataloader: Optional[data.DataLoader] = None,
    ):
        """Fit the classifier on the training data embedding.

        Parameters
        ----------
        train_dataloader: torch.utils.data.DataLoader
            Training dataloader yielding batches in the form `(X, y)`
            used for further embedding and training of the probes:

            - `X` will be given to `transform_step` of the embedding model.
            - `y` must have shape `(n_samples,)`.

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

        X_task = X  # do not copy
        y_task = y

        self.scorer_ = check_scoring(
            estimator=self.classifier, scoring=self.classification_scoring
        )

        if self.allow_nan:
            y_task, valid_mask, indices = self._filter_nan_or_inf(y_task)
            X_task = X_task[valid_mask]

        self.n_splits_ = self.cv.get_n_splits(X_task, y_task)

        self.cv_results_ = cross_validate(
            estimator=self.classifier,
            X=X_task,
            y=y_task,
            scoring=self.classification_scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            return_indices=True,
            return_estimator=True,
        )

        if self.allow_nan:  # remap indices to the unfiltered data
            for split in ["train", "test"]:
                relative_indices = self.cv_results_["indices"][split]
                absolute_indices = tuple(
                    indices[rel_idx] for rel_idx in relative_indices
                )
                self.cv_results_["indices"][split] = absolute_indices

        # Refit the classifier on the entire dataset
        self.classifier_ = clone(self.classifier).fit(X_task, y_task)
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
        y_pred: torch.Tensor, shape `(n_samples,)`
            The predicted labels.

        """
        check_is_fitted(self)

        X, _ = self.extract_features(test_dataloader)
        X = check_array(
            X, ensure_2d=True, dtype="numeric", force_all_finite=True
        )
        y_pred = self.classifier_.predict(X)

        y_pred = torch.as_tensor(y_pred)
        return y_pred

    def score(self, test_dataloader: data.DataLoader):
        """Score the probes on the test dataset.

        Parameters
        ----------
        test_dataloader: torch.utils.data.DataLoader
            Testing dataloader yielding batches in the form `(X, y)`.
            `y` must have shape `(n_samples,)`.

        Returns
        ----------
        score: float or dict
            Score over the test data. If multiple scoring metrics are provided
            for the classification, a dictionary with the metric name as key
            and score as value is returned.

        """
        check_is_fitted(self)
        X, y = self.extract_features(test_dataloader)
        X = check_array(
            X, ensure_2d=True, dtype="numeric", force_all_finite=True
        )
        y = self._check_y(y, force_all_finite=True)
        return self.scorer_(self.classifier_, X, y)

    @staticmethod
    def _check_y(y, force_all_finite: bool = True):
        """Check that y is a 1d array or that it can be casted to 1d."""
        y = check_array(y, ensure_2d=False, force_all_finite=force_all_finite)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(len(y))

        if y.ndim != 1:
            raise ValueError(f"Expected 1d array for `y`, got {y.ndim}d.")

        return y
