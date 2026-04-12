##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

from typing import Union

import torch
from sklearn.base import BaseEstimator as sk_BaseEstimator
from sklearn.metrics import check_scoring
from tqdm import tqdm

from nidl.estimators.base import BaseEstimator


class ModelProbe(BaseEstimator):
    """Estimator to probe the representation of an embedding estimator.

    It has the following logic:

    1) Embeds the input data (training+test) through the estimator using
       `transform_step` method (does not handle distributed multi-gpu forward
       pass).
    2) Train the probe on the training embedding with the `fit` method.
    3) Evaluate the probe on a dataset with the `predict` method.

    The metrics logged depend on the ``scoring`` parameter:

    - If a single score is provided, it logs ``test_score``.
    - If multiple scores are provided, it logs each score with its name
      (such as  ``test_accuracy``, ``test_auc``).

    Parameters
    ----------
    model: BaseEstimator
        The embedding estimator to be probed. It must implement the
        `transform_step` method that takes a batch of data and returns the
        corresponding embeddings.

    probe: sklearn.base.BaseEstimator
        The probe model to be trained on the embedding. It must
        implement `fit` and `predict` methods on numpy array.

    scoring: str, callable, list, tuple, or dict, default=None
        Strategy to evaluate the performance of the `probe` when calling the
        `score` method.

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

    prog_bar: bool, default=True
        Whether to display the metrics in the progress bar.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from nidl.estimators.probes import ModelProbe
    >>> callback = ModelProbe(
    ...     model=model,
    ...     probe=LogisticRegression(),
    ...     scoring=["accuracy", "balanced_accuracy"],
    ... )
    """

    def __init__(
        self,
        model: BaseEstimator,
        probe: sk_BaseEstimator,
        scoring: Union[str, callable, list, tuple, dict, None] = None,
        prog_bar: bool = True,
    ):
        super().__init__()
        self.model = model
        self.probe = probe
        self.scoring = scoring
        self.prog_bar = prog_bar

        self.scorer = check_scoring(self.probe, scoring=self.scoring)

    def fit(self, dataloader):
        """Fit the probe on the training data embeddings."""

        X_train, y_train = self.extract_features(dataloader)
        return self.probe.fit(X_train, y_train)

    def extract_features(self, dataloader):
        """Extract features from a dataloader with the BaseEstimator.

        By default, it uses the `transform_step` logic applied on each batch to
        get the embeddings with the labels.
        The input dataloader should yield batches of the form `(X, y)` where X
        is the input data and y is the label.

        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            The dataloader to extract features from. It should yield batches of
            the form `(X, y)` where `X` is the input data and `y` is the label.

        Returns
        -------
        tuple of (z, y)
            Tuple of numpy arrays (z, y) where z are the extracted features
            and y are the corresponding labels.

        """

        self.model.eval()
        X, y = [], []

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc="Extracting features",
                disable=not self.prog_bar,
                leave=False,
            ):
                x_batch, y_batch = batch
                x_batch = x_batch.to(self.model.device, non_blocking=True)
                y_batch = y_batch.to(self.model.device, non_blocking=True)
                features = self.model.transform_step(
                    x_batch, batch_idx=batch_idx
                )
                X.append(features.detach())
                y.append(y_batch.detach())
                del x_batch  # free memory

        # Concatenate the embeddings and move to numpy
        X = torch.cat(X).cpu().numpy()
        y = torch.cat(y).cpu().numpy()

        return X, y

    def predict(self, dataloader):
        """Predict with the probe on the test data embeddings."""

        X_test, _ = self.extract_features(dataloader)
        return self.probe.predict(X_test)

    def score(self, dataloader, scoring=None):
        """Evaluate the probe on the test data embeddings.

        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            The dataloader to extract features from. It should yield batches of
            the form `(X, y)` where `X` is the input data and `y` is the label.

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

        X_test, y_test = self.extract_features(dataloader)
        if scoring is not None:
            scorer = check_scoring(self.probe, scoring=scoring)
        else:
            scorer = self.scorer
        return scorer(self.probe, X_test, y_test)
