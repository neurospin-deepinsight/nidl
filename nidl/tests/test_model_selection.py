import numpy as np
import torch
import unittest
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
        b = x_batch.view(x_batch.shape[0], -1).float()
        mean = b.mean(dim=1, keepdim=True)
        std = b.std(dim=1, unbiased=False, keepdim=True)
        return torch.cat([mean, std], dim=1)


class TestMultiTaskProbingCV(unittest.TestCase):

    def test_parse_tasks_valid_and_invalid(self):
        est = SimpleEmbeddingEstimator()
        mt = MultiTaskProbingCV(estimator=est, tasks="classification")
        self.assertEqual(mt.tasks, "classification")

        mt = MultiTaskProbingCV(estimator=est, tasks=["classification", "regression"])
        self.assertEqual(mt.tasks, ["classification", "regression"])

        with self.assertRaises(ValueError):
            MultiTaskProbingCV(estimator=est, tasks="not_a_task")

        with self.assertRaises(TypeError):
            MultiTaskProbingCV(estimator=est, tasks=123)

    def test_parse_task_names_errors(self):
        est = SimpleEmbeddingEstimator()
        with self.assertRaises(ValueError):
            MultiTaskProbingCV(
                estimator=est,
                tasks=["classification", "regression"],
                task_names=["task", "task"],
            )

        with self.assertRaises(ValueError):
            MultiTaskProbingCV(
                estimator=est,
                tasks=["classification", "regression"],
                task_names=["only_one_name"],
            )

    def test_parse_cv_integer_and_invalid(self):
        est = SimpleEmbeddingEstimator()
        mt = MultiTaskProbingCV(estimator=est, tasks="classification", cv=3)
        self.assertTrue(hasattr(mt.cv, "split"))

        with self.assertRaises(TypeError):
            MultiTaskProbingCV(estimator=est, tasks="classification", cv=object())

    def test_parse_probes_invalid_types(self):
        est = SimpleEmbeddingEstimator()
        with self.assertRaises(TypeError):
            MultiTaskProbingCV(
                estimator=est, tasks="classification", classification_probe=Ridge()
            )

        with self.assertRaises(TypeError):
            MultiTaskProbingCV(
                estimator=est, tasks="regression", regression_probe=LogisticRegression()
            )

    def test_filter_nan_or_inf_and_check_y(self):
        est = SimpleEmbeddingEstimator()
        mt = MultiTaskProbingCV(estimator=est, tasks="classification")

        y1 = np.array([1.0, np.nan, 2.0])
        y_checked = mt._check_y(y1, force_all_finite=False)
        self.assertEqual(y_checked.shape, (3, 1))

        y_filtered, mask, indices = mt._filter_nan_or_inf(y1)
        self.assertTrue(np.array_equal(indices, np.array([0, 2])))
        self.assertEqual(mask.shape, (3,))
        self.assertEqual(y_filtered.shape, (2,))

        y2 = np.array([[1.0, 0.0], [np.nan, 3.0], [2.0, 4.0]])
        y_filtered2, mask2, indices2 = mt._filter_nan_or_inf(y2)
        self.assertEqual(mask2.sum(), 2)
        self.assertEqual(y_filtered2.shape, (2, 2))
        self.assertTrue(np.array_equal(indices2, np.array([0, 2])))

    def test_extract_features_train_state_restored(self):
        est = SimpleEmbeddingEstimator()
        est.train()
        self.assertTrue(est.training)
        mt = MultiTaskProbingCV(estimator=est, tasks="classification")

        X = torch.arange(12, dtype=torch.float32).view(6, 2)
        y = torch.arange(6)
        dl = make_dataloader(X, y, batch_size=3)

        X_out, y_out = mt.extract_features(dl)
        self.assertIsInstance(X_out, np.ndarray)
        self.assertEqual(X_out.shape[0], 6)
        self.assertEqual(y_out.shape[0], 6)
        self.assertTrue(est.training)

    def test_fit_predict_score_classification_and_regression(self):
        X_np, y_np = self.make_regression_and_classification_data(n_samples=40)
        X = torch.from_numpy(X_np)
        y_cls = y_np[:, 0].astype(np.int64)
        y_reg = y_np[:, 1].astype(np.float32)
        y_multi = np.stack([y_cls, y_reg], axis=1)
        y_tensor = torch.from_numpy(y_multi)

        est = SimpleEmbeddingEstimator()
        mt = MultiTaskProbingCV(
            estimator=est,
            tasks=["classification", "regression"],
            task_names=["cls", "reg"],
            cv=KFold(n_splits=3, shuffle=True, random_state=0),
            classification_probe=LogisticRegression(max_iter=1000),
            regression_probe=Ridge(),
            n_jobs=1,
        )

        train_dl = make_dataloader(X, y_tensor, batch_size=10)
        fitted = mt.fit(train_dl)
        self.assertIs(fitted, mt)
        self.assertTrue(mt.fitted_)
        self.assertIn("cls", mt.cv_results_)
        self.assertIn("reg", mt.cv_results_)
        self.assertIn("estimator", mt.cv_results_["cls"])
        self.assertIn("indices", mt.cv_results_["cls"])

        test_dl = make_dataloader(X, y_tensor, batch_size=8)
        y_pred = mt.predict(test_dl)
        self.assertIsInstance(y_pred, torch.Tensor)
        self.assertEqual(y_pred.shape, (X.shape[0], 2))

        scores = mt.score(test_dl)
        self.assertIsInstance(scores, dict)
        self.assertIn("cls", scores)
        self.assertIn("reg", scores)

    def test_fit_raises_on_unfitted_estimator(self):
        X = torch.arange(12, dtype=torch.float32).view(6, 2)
        y = torch.arange(6)
        dl = make_dataloader(X, y, batch_size=3)

        est = SimpleEmbeddingEstimator()
        est.fitted_ = False
        mt = MultiTaskProbingCV(
            estimator=est,
            tasks=["classification", "regression"],
            cv=3,
        )
        with self.assertRaises(Exception):
            mt.fit(dl)

    def test_score_raises_on_task_count_mismatch(self):
        X_np, y_np = self.make_regression_and_classification_data(n_samples=30)
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

        y_single = torch.from_numpy(y_np[:, 0].astype(np.int64))
        test_dl_wrong = make_dataloader(X, y_single, batch_size=10)

        with self.assertRaises(ValueError):
            mt.score(test_dl_wrong)

    def test_allow_nan_remaps_indices(self):
        n = 30
        X_np, y_np = self.make_regression_and_classification_data(n_samples=n)
        X = torch.from_numpy(X_np)
        y_multi = np.stack([y_np[:, 0].astype(np.float64), y_np[:, 1].astype(np.float64)], axis=1)
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

        idxs = mt.cv_results_["cls"]["indices"]
        train_indices = np.concatenate([np.asarray(a) for a in idxs["train"]]) if isinstance(idxs["train"], (list, tuple)) else np.asarray(idxs["train"])
        self.assertTrue(np.all((train_indices >= 0) & (train_indices < n)))

    def test_get_tasks_length_mismatch_raises(self):
        est = SimpleEmbeddingEstimator()
        mt = MultiTaskProbingCV(estimator=est, tasks="classification")
        tasks = mt._get_tasks(2)
        self.assertIsInstance(tasks, list)
        self.assertEqual(len(tasks), 2)

        mt_bad = MultiTaskProbingCV(estimator=est, tasks=["classification", "regression"])
        with self.assertRaises(ValueError):
            mt_bad._get_tasks(3)

    def test_get_task_names_length_mismatch_raises(self):
        est = SimpleEmbeddingEstimator()
        mt = MultiTaskProbingCV(estimator=est, tasks="classification", task_names=None)
        names = mt._get_task_names(3)
        self.assertEqual(len(names), 3)

        mt2 = MultiTaskProbingCV(estimator=est, tasks=["classification", "regression"], task_names=["a", "b"])
        with self.assertRaises(ValueError):
            mt2._get_task_names(3)

    def test_probe_estimators_fitted_on_entire_data(self):
        X_np, y_np = self.make_regression_and_classification_data(n_samples=40)
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

        test_dl = make_dataloader(X, y_tensor, batch_size=10)
        X_emb, _ = mt.extract_features(test_dl)
        for tn, probe in mt.probe_estimators_.items():
            preds = probe.predict(X_emb)
            self.assertEqual(len(preds), X_emb.shape[0])

    @staticmethod
    def make_regression_and_classification_data(n_samples=50, n_features=4, seed=0):
        rng = np.random.RandomState(seed)
        X = rng.randn(n_samples, n_features).astype(np.float32)
        cls = (X.sum(axis=1) > 0).astype(int)
        reg = (X[:, 0] * 2.0 + X[:, 1] * -1.0 + rng.randn(n_samples) * 0.1).astype(np.float32)
        y = np.stack([cls, reg], axis=1)
        return X, y


if __name__ == "__main__":
    unittest.main()
