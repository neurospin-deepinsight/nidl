# test_multitask_probing_cv.py
import math
import numpy as np
import torch
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold

from nidl.estimators.base import BaseEstimator, TransformerMixin
from nidl.model_selection import MultiTaskProbingCV


# ------------------------
# Helpers for tests
# ------------------------
def make_dataloader(X: torch.Tensor, y: torch.Tensor, batch_size: int = 8):
    """Simple DataLoader-like generator that yields (x_batch, y_batch)."""
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


class SimpleEmbeddingEstimator(TransformerMixin, BaseEstimator):
    """
        Minimal embedding estimator that returns a deterministic low-dimensional
        embedding for testing purposes. It simulates a model that has been fitted
        and can transform input data into a 2D embedding.
    """

    def __init__(self):
        super().__init__()
        self.fitted_ = True

    def transform_step(
            self, x_batch: torch.Tensor, 
            batch_idx: int = 0, dataloader_idx: int = 0
        ):
        # return a deterministic low-dimensional embedding
        # shape: (batch_size, 2)
        # Use both content and batch_idx so it's easy to test
        b = x_batch.view(x_batch.shape[0], -1).float()
        # Create two features: mean and std per sample (kept deterministic)
        mean = b.mean(dim=1, keepdim=True)
        std = b.std(dim=1, unbiased=False, keepdim=True)
        return torch.cat([mean, std], dim=1)


# ------------------------
# Tests: parsing helpers and validation
# ------------------------
def test_parse_tasks_valid_and_invalid():
    # valid string
    est = SimpleEmbeddingEstimator()
    mt = MultiTaskProbingCV(estimator=est, tasks="classification")
    assert mt.tasks == "classification"

    # valid list
    mt = MultiTaskProbingCV(estimator=est, tasks=["classification", "regression"])
    assert mt.tasks == ["classification", "regression"]

    # invalid string
    with pytest.raises(ValueError):
        MultiTaskProbingCV(estimator=est, tasks="not_a_task")

    # invalid type
    with pytest.raises(TypeError):
        MultiTaskProbingCV(estimator=est, tasks=123)


def test_parse_task_names_errors():
    est = SimpleEmbeddingEstimator()

    # non-unique task names
    with pytest.raises(ValueError):
        MultiTaskProbingCV(
            estimator=est,
            tasks=["classification", "regression"],
            task_names=["task", "task"],
        )

    # mismatch lengths between task_names and tasks
    with pytest.raises(ValueError):
        MultiTaskProbingCV(
            estimator=est,
            tasks=["classification", "regression"],
            task_names=["only_one_name"],
        )


def test_parse_cv_integer_and_invalid():
    est = SimpleEmbeddingEstimator()

    # valid integer -> check_cv accepted
    mt = MultiTaskProbingCV(estimator=est, tasks="classification", cv=3)
    # check_cv converts to splitter, so there should be an object with split method
    assert hasattr(mt.cv, "split")

    # invalid cv type
    with pytest.raises(TypeError):
        MultiTaskProbingCV(estimator=est, tasks="classification", cv=object())


def test_parse_probes_invalid_types():
    est = SimpleEmbeddingEstimator()

    # classification probe must be a classifier or None
    with pytest.raises(TypeError):
        MultiTaskProbingCV(
            estimator=est, tasks="classification", classification_probe=Ridge()
        )

    # regression probe must be a regressor or None
    with pytest.raises(TypeError):
        MultiTaskProbingCV(
            estimator=est, tasks="regression", regression_probe=LogisticRegression()
        )


# ------------------------
# Tests: internal utilities
# ------------------------
def test_filter_nan_or_inf_and_check_y():
    est = SimpleEmbeddingEstimator()
    mt = MultiTaskProbingCV(estimator=est, tasks="classification")

    # 1D y becomes 2D and NaN handling
    y1 = np.array([1.0, np.nan, 2.0])
    y_checked = mt._check_y(y1, force_all_finite=False)
    assert y_checked.shape == (3, 1)

    # check _filter_nan_or_inf 1d
    y_filtered, mask, indices = mt._filter_nan_or_inf(y1)
    assert np.array_equal(indices, np.array([0, 2]))
    assert mask.shape == (3,)
    assert y_filtered.shape == (2,)

    # 2D with NaN rows
    y2 = np.array([[1.0, 0.0], [np.nan, 3.0], [2.0, 4.0]])
    y_filtered2, mask2, indices2 = mt._filter_nan_or_inf(y2)
    assert mask2.sum() == 2
    assert y_filtered2.shape == (2, 2)
    assert np.array_equal(indices2, np.array([0, 2]))


# ------------------------
# Tests: extract_features preserves train/eval state and returns arrays
# ------------------------
def test_extract_features_train_state_restored():
    est = SimpleEmbeddingEstimator()
    # put in training mode initially
    est.train()
    assert est.training is True
    mt = MultiTaskProbingCV(estimator=est, tasks="classification")

    # tiny dataset
    X = torch.arange(12, dtype=torch.float32).view(6, 2)
    y = torch.arange(6)
    dl = make_dataloader(X, y, batch_size=3)

    X_out, y_out = mt.extract_features(dl)
    # features should be numpy and shape (6,2)
    assert isinstance(X_out, np.ndarray)
    assert X_out.shape[0] == 6
    assert y_out.shape[0] == 6

    # estimator training state should be restored (it was training initially)
    assert est.training is True


# ------------------------
# Integration tests: fit/predict/score on synthetic data
# ------------------------
def make_regression_and_classification_data(n_samples=50, n_features=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    # classification targets: two classes based on sum of features
    cls = (X.sum(axis=1) > 0).astype(int)
    # regression targets: a noisy linear function
    reg = (X[:, 0] * 2.0 + X[:, 1] * -1.0 + rng.randn(n_samples) * 0.1).astype(
        np.float32
    )
    y = np.stack([cls, reg], axis=1)
    return X, y


def test_fit_predict_score_classification_and_regression(tmp_path):
    # Create data
    X_np, y_np = make_regression_and_classification_data(n_samples=40)
    X = torch.from_numpy(X_np)
    # ensure classification task is integral for sklearn classifiers
    y_cls = y_np[:, 0].astype(np.int64)
    y_reg = y_np[:, 1].astype(np.float32)

    # Multi-task y: [classification, regression]
    y_multi = np.stack([y_cls, y_reg], axis=1)
    y_tensor = torch.from_numpy(y_multi)

    est = SimpleEmbeddingEstimator()
    mt = MultiTaskProbingCV(
        estimator=est,
        tasks=["classification", "regression"],
        task_names=["cls", "reg"],
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        # encourage deterministic behavior of probes
        classification_probe=LogisticRegression(max_iter=1000),
        regression_probe=Ridge(),
        n_jobs=1,
    )

    train_dl = make_dataloader(X, y_tensor, batch_size=10)
    fitted = mt.fit(train_dl)
    assert fitted is mt
    assert mt.fitted_ is True

    # Check that cv_results_ populated for both tasks and indices exist
    assert "cls" in mt.cv_results_
    assert "reg" in mt.cv_results_
    assert "estimator" in mt.cv_results_["cls"]
    assert "indices" in mt.cv_results_["cls"]

    # Make test dataloader (same data here for simplicity)
    test_dl = make_dataloader(X, y_tensor, batch_size=8)
    y_pred = mt.predict(test_dl)
    # predictions shape: (n_samples, n_tasks)
    assert isinstance(y_pred, torch.Tensor)
    assert y_pred.shape == (X.shape[0], 2)

    # Score on the test dataset
    scores = mt.score(test_dl)
    # Should return a dict with per-task scores
    assert isinstance(scores, dict)
    assert "cls" in scores and "reg" in scores

def test_fit_raises_on_unfitted_estimator():
    # tiny dataset
    X = torch.arange(12, dtype=torch.float32).view(6, 2)
    y = torch.arange(6)
    dl = make_dataloader(X, y, batch_size=3)

    est = SimpleEmbeddingEstimator() 
    est.fitted_ = False  # Simulate unfitted state
    mt = MultiTaskProbingCV(
        estimator=est,
        tasks=["classification", "regression"],
        cv=3,
    )

    with pytest.raises(Exception):
        mt.fit(dl)

def test_score_raises_on_task_count_mismatch():
    X_np, y_np = make_regression_and_classification_data(n_samples=30)
    X = torch.from_numpy(X_np)
    y_multi = np.stack([y_np[:, 0].astype(np.int64), y_np[:, 1].astype(np.float32)], axis=1)
    y_tensor = torch.from_numpy(y_multi)

    est = SimpleEmbeddingEstimator()
    mt = MultiTaskProbingCV(
        estimator=est,
        tasks=["classification", "regression"],
        cv=3,
    )

    train_dl = make_dataloader(X, y_tensor, batch_size=10)
    mt.fit(train_dl)

    # Create test dataloader with wrong number of tasks (single-target)
    y_single = torch.from_numpy(y_np[:, 0].astype(np.int64))
    test_dl_wrong = make_dataloader(X, y_single, batch_size=10)

    with pytest.raises(ValueError):
        mt.score(test_dl_wrong)


# ------------------------
# Tests: allow_nan behavior and indices remapping
# ------------------------
def test_allow_nan_remaps_indices():
    # Create data with NaN in classification labels
    n = 30
    X_np, y_np = make_regression_and_classification_data(n_samples=n)
    X = torch.from_numpy(X_np)
    # Put a few NaNs in the classification column
    y_multi = np.stack([y_np[:, 0].astype(np.float64), y_np[:, 1].astype(np.float64)], axis=1)
    y_multi = y_multi.astype(np.float64)
    y_multi[2, 0] = np.nan
    y_multi[5, 0] = np.nan
    y_tensor = torch.from_numpy(y_multi)

    est = SimpleEmbeddingEstimator()
    mt = MultiTaskProbingCV(
        estimator=est,
        tasks=["classification", "regression"],
        task_names=["cls", "reg"],
        allow_nan=True,
        cv=KFold(n_splits=4, shuffle=True, random_state=0),
    )

    train_dl = make_dataloader(X, y_tensor, batch_size=7)
    mt.fit(train_dl)

    # For the classification task, cv_results_ indices should be remapped
    idxs = mt.cv_results_["cls"]["indices"]
    # indices should be numpy arrays of absolute indices in original data (length <= n)
    assert isinstance(idxs["train"], tuple) or isinstance(idxs["train"], np.ndarray) or isinstance(idxs["train"], list)
    # Every index must be in [0, n)
    # Cross-validate may return arrays or tuples depending on sklearn version; check contents generically:
    train_indices = np.concatenate([np.asarray(a) for a in idxs["train"]]) if isinstance(idxs["train"], (list, tuple)) else np.asarray(idxs["train"])
    assert np.all((train_indices >= 0) & (train_indices < n))


# ------------------------
# Edge/Error cases for internal getters
# ------------------------
def test_get_tasks_length_mismatch_raises():
    est = SimpleEmbeddingEstimator()
    mt = MultiTaskProbingCV(estimator=est, tasks="classification")
    # ask for 2 tasks while tasks is a string (but _get_tasks expects to replicate string)
    # Should return list of 2 elements when tasks is string
    tasks = mt._get_tasks(2)
    assert isinstance(tasks, list)
    assert len(tasks) == 2

    # but when tasks is a list with wrong length, should raise
    mt_bad = MultiTaskProbingCV(estimator=est, tasks=["classification", "regression"])
    with pytest.raises(ValueError):
        mt_bad._get_tasks(3)


def test_get_task_names_length_mismatch_raises():
    est = SimpleEmbeddingEstimator()
    # None -> auto-generated names
    mt = MultiTaskProbingCV(estimator=est, tasks="classification", task_names=None)
    names = mt._get_task_names(3)
    assert len(names) == 3

    # if provided but wrong length -> raise
    mt2 = MultiTaskProbingCV(estimator=est, tasks=["classification", "regression"], task_names=["a", "b"])
    with pytest.raises(ValueError):
        mt2._get_task_names(3)


# ------------------------
# Final sanity check: predictors are fitted on full dataset after fit
# ------------------------
def test_probe_estimators_fitted_on_entire_data():
    X_np, y_np = make_regression_and_classification_data(n_samples=40)
    X = torch.from_numpy(X_np)
    y_multi = np.stack([y_np[:, 0].astype(np.int64), y_np[:, 1].astype(np.float32)], axis=1)
    y_tensor = torch.from_numpy(y_multi)

    est = SimpleEmbeddingEstimator()
    mt = MultiTaskProbingCV(
        estimator=est,
        tasks=["classification", "regression"],
        task_names=["cls", "reg"],
        classification_probe=LogisticRegression(max_iter=1000),
        regression_probe=Ridge(),
        cv=3,
    )

    train_dl = make_dataloader(X, y_tensor, batch_size=8)
    mt.fit(train_dl)

    # The stored probe_estimators_ should be able to predict on the embeddings
    # create extraction of the embeddings and run predict
    test_dl = make_dataloader(X, y_tensor, batch_size=10)
    X_emb, _ = mt.extract_features(test_dl)
    # predict for each probe
    for tn, probe in mt.probe_estimators_.items():
        preds = probe.predict(X_emb)
        # length should match n_samples
        assert len(preds) == X_emb.shape[0]
