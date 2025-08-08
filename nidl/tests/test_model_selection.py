import numpy as np
import torch
import unittest
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from nidl.estimators.base import BaseEstimator, TransformerMixin
from nidl.model_selection import MultiTaskProbingCV, RegressionProbingCV, ClassificationProbingCV



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

    @staticmethod
    def make_dataloader(X: torch.Tensor, y: torch.Tensor, batch_size: int = 8):
        dataset = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    @staticmethod
    def make_regression_and_classification_data(n_samples=50, n_features=4, seed=0):
        rng = np.random.RandomState(seed)
        X = rng.randn(n_samples, n_features).astype(np.float32)
        cls = (X.sum(axis=1) > 0).astype(int)
        reg = (X[:, 0] * 2.0 + X[:, 1] * -1.0 + rng.randn(n_samples) * 0.1).astype(np.float32)
        y = np.stack([cls, reg], axis=1)
        return X, y

    def setUp(self):
        self.estimator = SimpleEmbeddingEstimator()
        self.classifier = LogisticRegression(max_iter=1000)
        self.regressor = Ridge()
        self.cv = KFold(n_splits=3, shuffle=True, random_state=0)
        self.probing = MultiTaskProbingCV(
            estimator=self.estimator,
            tasks=["classification", "regression"],
            task_names=["cls", "reg"],
            classifier=self.classifier,
            regressor=self.regressor,
            cv=self.cv,
            n_jobs=1,
            allow_nan=True
        )

    def test_parse_tasks_valid_and_invalid(self):
        mt = MultiTaskProbingCV(estimator=self.estimator, 
                                tasks="classification")
        self.assertEqual(mt.tasks, "classification")

        mt = MultiTaskProbingCV(estimator=self.estimator, 
                                tasks=["classification", "regression"])
        self.assertEqual(mt.tasks, ["classification", "regression"])

        with self.assertRaises(ValueError):
            MultiTaskProbingCV(estimator=self.estimator, tasks="not_a_task")

        with self.assertRaises(TypeError):
            MultiTaskProbingCV(estimator=self.estimator, tasks=123)

    def test_parse_task_names_errors(self):
        with self.assertRaises(ValueError):
            MultiTaskProbingCV(
                estimator=self.estimator,
                tasks=["classification", "regression"],
                task_names=["task", "task"],
            )

        with self.assertRaises(ValueError):
            MultiTaskProbingCV(
                estimator=self.estimator,
                tasks=["classification", "regression"],
                task_names=["only_one_name"],
            )

    def test_parse_cv_integer_and_invalid(self):
        mt = MultiTaskProbingCV(estimator=self.estimator, 
                                tasks="classification", cv=3)
        self.assertTrue(hasattr(mt.cv, "split"))

        with self.assertRaises(TypeError):
            MultiTaskProbingCV(estimator=self.estimator, 
                               tasks="classification", cv=object())

    def test_parse_probes_invalid_types(self):
        with self.assertRaises(TypeError):
            MultiTaskProbingCV(
                estimator=self.estimator, 
                tasks="classification", 
                classifier=self.regressor
            )

        with self.assertRaises(TypeError):
            MultiTaskProbingCV(
                estimator=self.estimator, 
                tasks="regression", 
                regressor=self.classifier
            )

    def test_filter_nan_or_inf_and_check_y(self):
        mt = MultiTaskProbingCV(estimator=self.estimator, tasks="classification")

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
        self.estimator.train()
        self.assertTrue(self.estimator.training)
        mt = MultiTaskProbingCV(estimator=self.estimator,
                                tasks="classification")

        X = torch.arange(12, dtype=torch.float32).view(6, 2)
        y = torch.arange(6)
        dl = self.make_dataloader(X, y, batch_size=3)

        X_out, y_out = mt.extract_features(dl)
        self.assertIsInstance(X_out, np.ndarray)
        self.assertEqual(X_out.shape[0], 6)
        self.assertEqual(y_out.shape[0], 6)
        self.assertTrue(self.estimator.training)

    def test_fit_predict_score_classification_and_regression(self):
        X_np, y_np = self.make_regression_and_classification_data(n_samples=40)
        X = torch.from_numpy(X_np)
        y_cls = y_np[:, 0].astype(np.int64)
        y_reg = y_np[:, 1].astype(np.float32)
        y_multi = np.stack([y_cls, y_reg], axis=1)
        y_tensor = torch.from_numpy(y_multi)
        mt = self.probing
        train_dl = self.make_dataloader(X, y_tensor, batch_size=10)
        fitted = mt.fit(train_dl)
        self.assertIs(fitted, mt)
        self.assertTrue(mt.fitted_)
        self.assertIn("cls", mt.cv_results_)
        self.assertIn("reg", mt.cv_results_)
        self.assertIn("estimator", mt.cv_results_["cls"])
        self.assertIn("indices", mt.cv_results_["cls"])

        test_dl = self.make_dataloader(X, y_tensor, batch_size=8)
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
        dl = self.make_dataloader(X, y, batch_size=3)

        self.estimator.fitted_ = False
        with self.assertRaises(Exception):
            self.probing.fit(dl)

    def test_score_raises_on_task_count_mismatch(self):
        X_np, y_np = self.make_regression_and_classification_data(n_samples=30)
        X = torch.from_numpy(X_np)
        y_multi = np.stack([y_np[:, 0].astype(np.int64), y_np[:, 1].astype(np.float32)], axis=1)
        y_tensor = torch.from_numpy(y_multi)

        train_dl = self.make_dataloader(X, y_tensor, batch_size=10)
        self.probing.fit(train_dl)

        y_single = torch.from_numpy(y_np[:, 0].astype(np.int64))
        test_dl_wrong = self.make_dataloader(X, y_single, batch_size=10)

        with self.assertRaises(ValueError):
            self.probing.score(test_dl_wrong)

    def test_allow_nan_remaps_indices(self):
        n = 30
        X_np, y_np = self.make_regression_and_classification_data(n_samples=n)
        X = torch.from_numpy(X_np)
        y_multi = np.stack([y_np[:, 0].astype(np.float64), y_np[:, 1].astype(np.float64)], axis=1)
        y_multi[2, 0] = np.nan
        y_multi[5, 0] = np.nan
        y_tensor = torch.from_numpy(y_multi)

        mt = MultiTaskProbingCV(
            estimator=self.estimator,
            tasks=["classification", "regression"],
            task_names=["cls", "reg"],
            allow_nan=True,
            cv=KFold(n_splits=4, shuffle=True, random_state=0),
        )

        train_dl = self.make_dataloader(X, y_tensor, batch_size=7)
        mt.fit(train_dl)

        idxs = mt.cv_results_["cls"]["indices"]
        train_indices = np.concatenate([np.asarray(a) for a in idxs["train"]])
        self.assertTrue(np.all((train_indices >= 0) & (train_indices < n)))

    def test_get_tasks_length_mismatch_raises(self):
        mt = MultiTaskProbingCV(estimator=self.estimator, 
                                tasks="classification")
        tasks = mt._get_tasks(2)
        self.assertIsInstance(tasks, list)
        self.assertEqual(len(tasks), 2)

        mt_bad = MultiTaskProbingCV(estimator=self.estimator, 
                                    tasks=["classification", "regression"])
        with self.assertRaises(ValueError):
            mt_bad._get_tasks(3)

    def test_get_task_names_length_mismatch_raises(self):
        mt = MultiTaskProbingCV(estimator=self.estimator, 
                                tasks="classification", task_names=None)
        names = mt._get_task_names(3)
        self.assertEqual(len(names), 3)

        mt2 = MultiTaskProbingCV(estimator=self.estimator, 
                                 tasks=["classification", "regression"], 
                                 task_names=["a", "b"])
        with self.assertRaises(ValueError):
            mt2._get_task_names(3)

    def test_probe_estimators_fitted_on_entire_data(self):
        X_np, y_np = self.make_regression_and_classification_data(n_samples=40)
        X = torch.from_numpy(X_np)
        y_multi = np.stack([y_np[:, 0].astype(np.int64), y_np[:, 1].astype(np.float32)], axis=1)
        y_tensor = torch.from_numpy(y_multi)

        train_dl = self.make_dataloader(X, y_tensor, batch_size=8)
        self.probing.fit(train_dl)

        test_dl = self.make_dataloader(X, y_tensor, batch_size=10)
        X_emb, _ = self.probing.extract_features(test_dl)
        for tn, probe in self.probing.probe_estimators_.items():
            preds = probe.predict(X_emb)
            self.assertEqual(len(preds), X_emb.shape[0])


class TestRegressionProbingCV(unittest.TestCase):

    def create_dataloader(self, n_samples=100, n_features=10, batch_size=20, with_nan=False):
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randn(n_samples).astype(np.float32)
        if with_nan:
            y[0] = np.nan
        dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
        return DataLoader(dataset, batch_size=batch_size)

    def setUp(self):
        self.estimator = SimpleEmbeddingEstimator()
        self.regressor = LinearRegression()
        self.cv = KFold(n_splits=3)
        self.probing = RegressionProbingCV(
            estimator=self.estimator,
            regressor=self.regressor,
            cv=self.cv,
            allow_nan=True
        )

    def test_fit_sets_attributes(self):
        train_loader = self.create_dataloader()
        self.probing.fit(train_loader)
        self.assertTrue(self.probing.fitted_)
        self.assertTrue(hasattr(self.probing, "regressor_"))
        self.assertTrue(hasattr(self.probing, "cv_results_"))
        self.assertEqual(self.probing.n_splits_, 3)

    def test_predict_returns_tensor(self):
        train_loader = self.create_dataloader()
        self.probing.fit(train_loader)
        test_loader = self.create_dataloader()
        preds = self.probing.predict(test_loader)
        self.assertIsInstance(preds, torch.Tensor)
        self.assertEqual(preds.ndim, 1)
        self.assertEqual(preds.shape[0], len(test_loader.dataset))

    def test_score_returns_float(self):
        train_loader = self.create_dataloader()
        self.probing.fit(train_loader)
        test_loader = self.create_dataloader()
        score = self.probing.score(test_loader)
        self.assertIsInstance(score, float)

    def test_fit_removes_nan_labels_when_allowed(self):
        train_loader = self.create_dataloader(with_nan=True)
        self.probing.allow_nan = True
        self.probing.fit(train_loader)

        # Get all train and test indices from cv_results_ 
        train_indices = np.concatenate(self.probing.cv_results_["indices"]["train"])
        test_indices = np.concatenate(self.probing.cv_results_["indices"]["test"])
        all_indices = np.concatenate([train_indices, test_indices])

        # Find indices of NaN labels in the original dataset
        y_all = train_loader.dataset.tensors[1].numpy()
        nan_indices = np.where(np.isnan(y_all))[0]

        # Assert none of the NaN indices are in any train or test split
        for nan_idx in nan_indices:
            self.assertNotIn(nan_idx, all_indices)

    def test_fit_raises_without_nan_allowed(self):
        train_loader = self.create_dataloader(with_nan=True)
        self.probing.allow_nan = False
        with self.assertRaises(ValueError):
            self.probing.fit(train_loader)

    def test_check_y_raises_on_wrong_shape(self):
        # y 2D with shape != (n_samples,)
        y_bad = np.array([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            self.probing._check_y(y_bad)


class TestClassificationProbingCV(unittest.TestCase):

    def create_dataloader(self, n_samples=100, n_features=10, n_classes=2,
                          batch_size=20, with_nan=False):
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, n_informative=5,
            n_classes=n_classes, random_state=42
        )
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        # Insert NaN for the first label
        if with_nan:
            y[0] = np.nan
        dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def setUp(self):
        self.estimator = SimpleEmbeddingEstimator()
        self.classifier = LogisticRegression(max_iter=200)
        self.probing = ClassificationProbingCV(
            estimator=self.estimator,
            classifier=self.classifier,
            cv=3,
            allow_nan=False
        )
        self.train_loader = self.create_dataloader()
        self.test_loader = self.create_dataloader()

    def test_fit(self):
        self.probing.fit(self.train_loader)
        self.assertTrue(hasattr(self.probing, "classifier_"))
        self.assertTrue(self.probing.fitted_)

    def test_predict(self):
        self.probing.fit(self.train_loader)
        y_pred = self.probing.predict(self.test_loader)
        self.assertIsInstance(y_pred, torch.Tensor)
        self.assertEqual(y_pred.shape[0], len(self.test_loader.dataset))
    
    def test_fit_predict_score(self):
        # Basic test of fit, predict, score with default logistic regression
        probe = ClassificationProbingCV(estimator=self.estimator, cv=KFold(3))
        probe.fit(self.train_loader)

        # Predict returns torch tensor of correct shape
        y_pred = probe.predict(self.test_loader)
        self.assertIsInstance(y_pred, torch.Tensor)
        self.assertEqual(y_pred.shape[0], len(self.test_loader.dataset))

        # Score returns a float
        score = probe.score(self.test_loader)
        self.assertIsInstance(score, float)
        self.assertLessEqual(score, 1)
    
    def test_fit_with_allow_nan_removes_nan(self):
        dataloader_nan =  self.create_dataloader(with_nan=True)

        probe = ClassificationProbingCV(estimator=self.estimator, 
                                        cv=KFold(3), 
                                        allow_nan=True)
        probe.fit(dataloader_nan)

        # Check nan index is removed from train/test indices in cv_results_
        for split in ("train", "test"):
            for split_indices in probe.cv_results_["indices"][split]:
                self.assertNotIn(0, split_indices)  # index 0 had nan

    def test_check_y_valid_and_invalid(self):
        # Test _check_y reshaping 2D (n, 1) array to 1D
        y_2d = np.ones((3, 1))
        y_checked = ClassificationProbingCV._check_y(y_2d)
        self.assertEqual(y_checked.shape, (len(y_2d),))

        # Test _check_y raises on wrong ndim
        y_bad = np.ones((2, 3))
        with self.assertRaises(ValueError):
            ClassificationProbingCV._check_y(y_bad)

    def test_score_with_multiple_metrics(self):
        # Use multiple scorers
        scorers = {
            'accuracy': make_scorer(accuracy_score),
            'neg_accuracy': make_scorer(accuracy_score, greater_is_better=False)
        }
        probe = ClassificationProbingCV(estimator=self.estimator, cv=KFold(3), 
                                        scoring=scorers)
        probe.fit(self.train_loader)

        score = probe.score(self.test_loader)
        # If multiple scorers, score is dict
        self.assertIsInstance(score, dict)
        self.assertIn('accuracy', score)
        self.assertIn('neg_accuracy', score)

    def test_predict_before_fit_raises(self):
        probe = ClassificationProbingCV(estimator=self.estimator)
        with self.assertRaises(Exception):
            probe.predict(self.dataloader)

    def test_score_before_fit_raises(self):
        probe = ClassificationProbingCV(estimator=self.estimator)
        with self.assertRaises(Exception):
            probe.score(self.dataloader)

    def test_check_y_shape(self):
        # test _check_y with 2D shape (n_samples, 1)
        y = np.array([[0], [1], [0], [1]])
        y_checked = self.probing._check_y(y)
        self.assertEqual(y_checked.ndim, 1)
        self.assertEqual(len(y_checked), y.size)

        # test _check_y with invalid shape
        y_bad = np.array([[0, 1], [1, 0]])
        with self.assertRaises(ValueError):
            self.probing._check_y(y_bad)


if __name__ == "__main__":
    unittest.main()
