##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

from typing import Union

import pytorch_lightning as pl
import torch
from sklearn.base import BaseEstimator as sk_BaseEstimator
from sklearn.model_selection import cross_validate
from sklearn.utils.validation import check_array
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from nidl.estimators.base import BaseEstimator
from nidl.utils.validation import _estimator_is


class ModelProbingCV(pl.Callback):
    """Callback to probe the representation of an embedding estimator on a
    dataset using cross-validation.

    It has the following logic:

    1) Embeds the input data through the estimator using `transform_step`.
    2) For each CV split, train the probe on the training embedding split and
       make predictions on the test embedding split.
    3) Compute and log the scores computed between the true and predicted
       targets for each CV split.

    The probing can be performed at the end of training epochs, validation
    epochs, and/or at the start/end of the test epoch.

    The metrics logged depend on the `scoring` parameter:

    - If a single score is provided, it logs ``fold{i}/test_score`` for each
      fold ``i``.
    - If multiple scores are provided, it logs each score with its name, such
      as ``fold{i}/test_accuracy`` or ``fold{i}/test_auc`` for each fold ``i``.

    Eventually, a ``prefix_score`` can be added to the score names when
    logging, such as ``ridge_`` or ``logreg_`` (giving
    ``fold{i}/ridge_test_accuracy`` or ``fold{i}/logreg_test_accuracy``).

    Parameters
    ----------
    dataloader: torch.utils.data.DataLoader
        Dataloader yielding batches in the form `(X, y)`
        for further embedding and cross-validation of the probe.

    probe: sklearn.base.BaseEstimator
        The probe model to be trained on the embedding. It must
        implement `fit` and `predict` methods.

    scoring: str, callable, list, tuple, or dict, default=None
        Strategy to evaluate the performance of the `probe` across
        cross-validation splits. The scores are logged into the
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

    cv: int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the probe is a classifier and ``y`` is
        either binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used. In all
        other cases, :class:`~sklearn.model_selection.KFold` is used.
        These splitters are instantiated with `shuffle=False` so the splits
        will be the same across calls.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the probe and computing
        the score are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a ``joblib.parallel_backend`` context.
        ``-1`` means using all processors.

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

    prefix_score: str, default=""
        Prefix to add to the score name when logging, such as ``ridge_`` or
        ``logreg_``.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        probe: sk_BaseEstimator,
        scoring: Union[str, callable, list, tuple, dict, None] = None,
        cv=None,
        n_jobs: Union[int, None] = None,
        every_n_train_epochs: Union[int, None] = 1,
        every_n_val_epochs: Union[int, None] = None,
        on_test_epoch_start: bool = False,
        on_test_epoch_end: bool = False,
        prog_bar: bool = True,
        prefix_score: str = "",
    ):
        super().__init__()
        self.dataloader = dataloader
        self.probe = probe
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.every_n_train_epochs = every_n_train_epochs
        self.every_n_val_epochs = every_n_val_epochs
        self._on_test_epoch_start = on_test_epoch_start
        self._on_test_epoch_end = on_test_epoch_end
        self.prog_bar = prog_bar
        self.prefix_score = prefix_score
        self.counter_val_epochs = 0

    def cross_validate(self, X, y):
        """Cross-validate the probe on the data embeddings."""
        return cross_validate(
            self.probe,
            X,
            y,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            return_train_score=False,
            return_estimator=False,
            return_indices=False,
        )

    def log_metrics(self, pl_module, scores):
        """Log all scores + times from sklearn.cross_validate into PL."""
        for key, values in scores.items():
            if hasattr(values, "__len__"):
                # per-fold
                for i, v in enumerate(values):
                    pl_module.log(
                        f"fold{i}/{self.prefix_score}{key}",
                        float(v),
                        prog_bar=self.prog_bar,
                        sync_dist=True,
                    )

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
                timeout=dataloader.timeout,
                worker_init_fn=dataloader.worker_init_fn,
                multiprocessing_context=dataloader.multiprocessing_context,
                prefetch_factor=dataloader.prefetch_factor,
                persistent_workers=dataloader.persistent_workers,
                pin_memory_device=dataloader.pin_memory_device,
                in_order=dataloader.in_order,
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
        X, y = self.extract_features(trainer, pl_module, self.dataloader)

        # Check arrays
        X, y = self.check_array(X, y)

        # For efficiency, fit/score on rank 0 only
        scores = None
        if trainer.is_global_zero:
            # Cross-validate the probe and get the scores
            scores = self.cross_validate(X, y)

        if trainer.world_size > 1:
            # Broadcast scores to all ranks
            scores = trainer.strategy.broadcast(scores, src=0)

        # Log metrics
        self.log_metrics(pl_module, scores)

    def extract_features(self, trainer, pl_module, dataloader):
        """Extract features from a dataloader with the BaseEstimator.

        By default, it uses the `transform_step` logic applied on each batch to
        get the embeddings with the labels.
        The input dataloader should yield batches of the form `(X, y)` where X
        is the input data and y is the label.

        Parameters
        ----------
        trainer: pl.Trainer
            The pytorch-lightning trainer.
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

        dataloader = self.adapt_dataloader_for_ddp(dataloader, trainer)

        pl_module.eval()
        X, y = [], []

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc="Extracting features",
                disable=(not trainer.is_global_zero),
                leave=False,
            ):
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
        if pl_module.trainer.world_size > 1:
            X = X.reshape(X.shape[0] * X.shape[1], *X.shape[2:])
            y = y.reshape(y.shape[0] * y.shape[1], *y.shape[2:])

        if is_training:
            pl_module.train()

        return X, y

    def check_array(self, X, y):
        """Check the input arrays for cross-validation.

        Parameters
        ----------
        X: np.ndarray
            The input features.
        y: np.ndarray
            The input targets.

        Returns
        -------
        tuple of (X, y)
            The checked input features and targets.

        """
        X = check_array(X, ensure_2d=True)
        y = check_array(y, ensure_2d=False)

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
