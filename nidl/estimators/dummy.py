##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

from typing import Any, Optional

import torch

from nidl.estimators.base import BaseEstimator, TransformerMixin


class DummyEmbeddingEstimator(TransformerMixin, BaseEstimator):
    """A dummy embedding estimator returning an embedding independent of the
    input data.

    Parameters
    ----------
    strategy: str, default="normal"
        The strategy to generate the dummy embedding with shape
        `(n_samples, n_features)`. It can be one of the following:

        * "identity": the embedding is the same as the input batch (flattened).
        * "normal": the embedding is generated from a normal distribution.
        * "uniform": the embedding is generated from a uniform distribution.
        * "constant": the embedding is a constant value.

    n_features: int, default=10
        The number of features in the embedding.

    constant: float, default=0.0
        The constant value to use when `strategy` is "constant".

    random_state: int, RandomState instance or None, default=None
        Controls the randomness of the embedding generation when `strategy` is
        "normal" or "uniform".
    """

    def __init__(
        self,
        strategy: str = "normal",
        n_features: int = 10,
        random_state: Optional[int] = None,
        constant: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.strategy = strategy
        self.n_features = n_features
        self.random_state = random_state
        self.constant = constant
        self.fitted_ = True  # No fitting needed for this dummy estimator

    def training_step(self, *args, **kwargs):
        """No training needed for this dummy estimator."""
        return torch.zeros((), device=self.device, requires_grad=True)

    def test_step(self, *args, **kwargs):
        """No testing needed for this dummy estimator."""
        return

    def configure_optimizers(self):
        """No optimizers needed for this dummy estimator."""
        return

    def transform_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = 0
    ):
        """Generates a dummy embedding with the same length as the input
        batch."""

        batch_size = len(batch)
        g = torch.Generator()
        if self.random_state is not None:
            g.manual_seed(self.random_state)
        if self.strategy == "identity":
            # Flatten the input batch to create the embedding
            embedding = batch.view(batch_size, -1)
        elif self.strategy == "normal":
            embedding = torch.randn(batch_size, self.n_features, generator=g)
        elif self.strategy == "uniform":
            embedding = torch.rand(batch_size, self.n_features, generator=g)
        elif self.strategy == "constant":
            if self.constant is None:
                raise ValueError(
                    "Constant target value has to be specified "
                    "when the constant strategy is used."
                )
            embedding = torch.full(
                (batch_size, self.n_features), fill_value=self.constant
            )
        else:
            raise ValueError(
                f"Invalid strategy '{self.strategy}'. "
                f"Expected one of 'normal', 'uniform', or 'constant'."
            )
        embedding = embedding.to(self.device)
        return embedding
