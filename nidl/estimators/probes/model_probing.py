##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

from typing import Optional, Union

import sklearn
import torch
from nibabel import data
from sklearn.metrics import check_scoring

from nidl.estimators.base import BaseEstimator
from nidl.utils.validation import _estimator_is, check_is_fitted


class ModelProbing(BaseEstimator):
    """Estimator to probe the representation of an embedding estimator.

    It has the following logic during `fit`:

    1) Embeds the training data through the embedding estimator (handles
       multi-gpu foward pass).
    2) Fit the probe on the training embedding (handles multi-cpu training).

    Then, `score` and `predict` methods evaluate the probe on a dataset with
    the same logic.

    Parameters
    ----------
    embedding_estimator: BaseEstimator
        The estimator to be probed. It must implement the `transform_step`
        method that takes a batch of data `X` and returns the corresponding
        embeddings.

    probe: sklearn.base.BaseEstimator
        The probe model to be trained on the embedding. It must
        implement `fit` and `predict` methods on numpy array.

    scoring: str, callable, list, tuple, or dict, default=None
        Strategy to evaluate the performance of the `probe` when calling the
        `score` method on this estimator.

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

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from nidl.estimators.probes import ModelProbing
    >>> from nidl.dummy import DummyEmbeddingEstimator
    >>> probing = ModelProbing(
    ...     embedding_estimator=DummyEmbeddingEstimator(),
    ...     probe=LogisticRegression(),
    ...     scoring=["accuracy", "balanced_accuracy"],
    ... )
    >>> probing.fit(train_dataloader)
    ModelProbing(...)
    >>> probing.score(test_dataloader)
    {'accuracy': 0.85, 'balanced_accuracy': 0.83}
    """

    def __init__(
        self,
        embedding_estimator: BaseEstimator,
        probe: sklearn.base.BaseEstimator,
        scoring: Union[str, callable, list, tuple, dict, None] = None,
        **kwargs,
    ):
        ignore = kwargs.pop("ignore", [])
        if "embedding_estimator" not in ignore:
            ignore.append("embedding_estimator")
        super().__init__(**kwargs, ignore=ignore)
        self.embedding_estimator = embedding_estimator
        self.probe = probe
        self.scoring = scoring
        self.scorer = check_scoring(self.probe, scoring=self.scoring)

        if not isinstance(
            embedding_estimator, BaseEstimator
        ) or not _estimator_is("transformer"):
            raise TypeError(
                "The embedding estimator must derive from 'BaseEstimator' and "
                f"'TransformerMixin' got {type(embedding_estimator)}"
            )
        if not isinstance(probe, sklearn.base.BaseEstimator):
            raise TypeError(
                f"The probe must be a sklearn estimator, got {type(probe)}"
            )

    def fit(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ):
        """Fit the probe on the training data embeddings.

        Parameters
        ----------
        train_dataloader: torch.utils.data.DataLoader
            Training dataloader yielding batches in the form `(X, y)`
            used for further embedding and training of the probes.

        val_dataloader: torch.utils.data.DataLoader or None, default=None
            Ignored.

        Returns
        ----------
        self: object
            The fitted estimator.
        """
        X_train, y_train = self.embedding_estimator.transform_with_targets(
            train_dataloader
        )

        # Fit the probe on the training embedding. Only the process with global
        # rank 0 fits the probe to avoid redundant computations.
        fitted_probe = None
        if self.embedding_estimator.trainer.is_global_zero:
            X_train = X_train.cpu().numpy()
            y_train = y_train.cpu().numpy()
            fitted_probe = self.probe.fit(X_train, y_train)
        # Broadcast the fitted sklearn object from rank 0 to all processes
        fitted_probe = self.embedding_estimator.trainer.strategy.broadcast(
            fitted_probe, src=0
        )
        self.probe = fitted_probe

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
        y_pred: torch.Tensor
            The predicted labels.

        """
        check_is_fitted(self)

        X, _ = self.embedding_estimator.transform_with_targets(test_dataloader)
        X = X.cpu().numpy()
        return self.probe.predict(X)

    def score(self, test_dataloader: data.DataLoader, scoring=None):
        """Score the probe on the test dataset.

        Parameters
        ----------
        test_dataloader: torch.utils.data.DataLoader
            Testing dataloader yielding batches in the form `(X, y)`.
            `y` must have same number of samples as `X`.

        scoring: str, callable, list, tuple, or dict, default=None
            Strategy to evaluate the performance of the `probe`. This allows to
            override the default `scoring` strategy defined at initialization.

            If `scoring` represents a single score, one can use:

            - a single string (see :ref:`scoring_string_names`);
            - a callable (see :ref:`scoring_callable`) that returns a single
              value.
            - `None`, the `probe`'s
            :ref:`default evaluation criterion <scoring_api_overview>` is used.

            If `scoring` represents multiple scores, one can use:

            - a list or tuple of unique strings;
            - a callable returning a dictionary where the keys are the metric
            names and the values are the metric scores;
            - a dictionary with metric names as keys and callables a values.

        Returns
        -------
        float or dict
            The score(s) of the probe on the data embeddings. If a single score
            is used in the scoring strategy, returns a float. If multiple
            scores are defined, returns a dictionary with metric names as keys
            and metric scores as values.
        """
        check_is_fitted(self)

        X_test, y_test = self.embedding_estimator.transform_with_targets(
            test_dataloader
        )
        X_test = X_test.cpu().numpy()
        y_test = y_test.cpu().numpy()

        if scoring is not None:
            scorer = check_scoring(self.probe, scoring=scoring)
        else:
            scorer = self.scorer
        return scorer(self.probe, X_test, y_test)
