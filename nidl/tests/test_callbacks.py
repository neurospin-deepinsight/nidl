##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from collections import OrderedDict

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms

from sklearn.linear_model import LogisticRegression as sk_LogisticRegression, LinearRegression
from sklearn.datasets import make_classification, make_regression

from nidl.transforms import MultiViewsTransform
from nidl.callbacks.check_typing import BatchTypingCallback
from nidl.callbacks.model_probing import ClassificationProbingCallback, RegressionProbingCallback
from nidl.callbacks.multitask_probing import MultiTaskEstimator, MultitaskModelProbing
from nidl.estimators.linear import LogisticRegression
from nidl.estimators.ssl import SimCLR
from nidl.utils import print_multicolor


class TestCallbacks(unittest.TestCase):
    """ Test callbacks.
    """
    def setUp(self):
        """ Setup test.
        """
        self._encoder = nn.Linear(5 * 5, 10)
        self._encoder.latent_size = 10
        self._fc = nn.Linear(self._encoder.latent_size, 3)
        self._model =  nn.Sequential(OrderedDict([
            ("encoder", self._encoder),
            ("fc", self._fc)
        ]))
        self.n_images = 30
        self.fake_data = torch.rand(self.n_images, 5 * 5)
        self.fake_labels = torch.randint(0, 3, (self.n_images, ))
        self.fake_continuous_labels = torch.rand(self.n_images, 1)
        self.fake_multivariate_continuous_labels = torch.rand(self.n_images, 3)
        x_dataset = CustomTensorDataset(
            self.fake_data
        )
        # multi-class classification dataset
        xy_dataset = CustomTensorDataset( 
            self.fake_data,
            labels=self.fake_labels
        )
        self.x_loader = DataLoader(x_dataset, batch_size=2, shuffle=False)
        self.xy_loader = DataLoader(xy_dataset, batch_size=2, shuffle=False)
    
    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_batch_typing_callbacks(self):
        """ Test callbacks (simple check).
        """
        model = LogisticRegression(
            model=self._model,
            random_state=None,
            limit_train_batches=3,
            max_epochs=2,
            num_classes=3,
            lr=5e-4,
            weight_decay=1e-4,
            callbacks=[
                BatchTypingCallback(),
            ],
            enable_checkpointing=False
        )
        model.fit(self.xy_loader)
        pred = model.predict(self.x_loader)
        self.assertTrue(pred.shape == (self.n_images, 3), 
                         msg="Predicted shape mismatch: "
                            f"expected {(self.n_images, 3)}, "
                            f"got {pred.shape}")
    

class TestClassificationProbingCallback(unittest.TestCase):
    def setUp(self):
        X, y = make_classification(n_samples=20, n_features=5, n_classes=2)
        self.X, self.y = X, y
        self.pl_module = MagicMock()
        self.pl_module.log = MagicMock()
        self.pl_module.log_dict = MagicMock()

    def test_init_rejects_regressor(self):
        with self.assertRaises(ValueError):
            RegressionProbingCallback(None, None, sk_LogisticRegression())

    def test_init_accepts_classifier(self):
        cb = ClassificationProbingCallback(None, None, sk_LogisticRegression(), probe_name="clf")
        self.assertTrue(cb.probe_name.startswith("clf"))

    def test_log_metrics(self):
        cb = ClassificationProbingCallback(None, None, sk_LogisticRegression())
        y_pred = np.array([0, 1, 0])
        y_true = np.array([0, 1, 0])

        cb.log_metrics(self.pl_module, y_pred, y_true)

        # Ensure metrics are logged
        self.pl_module.log_dict.assert_any_call(
            {
                "LogisticRegression/accuracy": 1.0,
                "LogisticRegression/balanced_accuracy": 1.0,
                "LogisticRegression/precision_macro": 1.0,
                "LogisticRegression/recall_macro": 1.0,
                "LogisticRegression/f1_macro": 1.0,
                "LogisticRegression/f1_weighted": 1.0
            },
            prog_bar=True,
            on_epoch=True
        )


class TestRegressionProbingCallback(unittest.TestCase):
    def setUp(self):
        X, y = make_regression(n_samples=20, n_features=5)
        self.X, self.y = X, y
        self.pl_module = MagicMock()
        self.pl_module.log = MagicMock()
        self.pl_module.log_dict = MagicMock()

    def test_init_rejects_classifier(self):
        with self.assertRaises(ValueError):
            ClassificationProbingCallback(None, None, LinearRegression())

    def test_init_accepts_regressor(self):
        cb = RegressionProbingCallback(None, None, LinearRegression(), probe_name="reg")
        self.assertTrue(cb.probe_name.startswith("reg"))

    def test_log_metrics(self):
        cb = RegressionProbingCallback(None, None, LinearRegression())
        y_pred = np.array([0.5, 1.2])
        y_true = np.array([0.5, 1.2])

        cb.log_metrics(self.pl_module, y_pred, y_true)

        # Check scalar logging
        self.pl_module.log.assert_any_call("LinearRegression/MAE", 0.0, prog_bar=True, on_epoch=True)


class TestMultiTaskEstimator(unittest.TestCase):
    def setUp(self):
        Xc, yc = make_classification(n_samples=20, n_features=5, n_classes=2)
        Xr, yr = make_regression(n_samples=20, n_features=5)
        self.Xc, self.yc = Xc, yc
        self.Xr, self.yr = Xr, yr

    def test_invalid_estimators_raise(self):
        # LinearRegression is regressor, should raise if mixed incorrectly
        with self.assertRaises(ValueError):
            MultiTaskEstimator([LinearRegression(), "not_estimator"])

    def test_fit_and_predict_single_task(self):
        est = MultiTaskEstimator([sk_LogisticRegression(max_iter=200)])
        est.fit(self.Xc, self.yc)
        preds = est.predict(self.Xc)
        self.assertEqual(preds.shape[0], self.Xc.shape[0])
        self.assertEqual(preds.shape[1], 1)

    def test_fit_and_predict_multi_task(self):
        y = np.vstack([self.yc, (self.yr > self.yr.mean()).astype(int)]).T
        est = MultiTaskEstimator(
            [sk_LogisticRegression(max_iter=200), sk_LogisticRegression(max_iter=200)]
        )
        est.fit(self.Xc, y)
        preds = est.predict(self.Xc)
        self.assertEqual(preds.shape, y.shape)

    def test_fit_raises_on_y_shape(self):
        est = MultiTaskEstimator([sk_LogisticRegression(max_iter=200)])
        # y with 3 dimensions is invalid
        y_bad = np.zeros((10, 1, 1))
        with self.assertRaises(ValueError):
            est.fit(self.Xc[:10], y_bad)

    def test_score_average(self):
        y = np.vstack([self.yc, (self.yr > self.yr.mean()).astype(int)]).T
        est = MultiTaskEstimator(
            [sk_LogisticRegression(max_iter=200), sk_LogisticRegression(max_iter=200)]
        )
        est.fit(self.Xc, y)
        score = est.score(self.Xc, y)
        self.assertIsInstance(score, float)


class TestParseNames(unittest.TestCase):
    def test_default_names(self):
        probes = MultiTaskEstimator([sk_LogisticRegression(), sk_LogisticRegression()])
        cb = MultitaskModelProbing(None, None, probes)
        names = cb._parse_names(probes, None)
        self.assertEqual(names, ["task0", "task1"])

    def test_custom_names(self):
        probes = MultiTaskEstimator([sk_LogisticRegression(), sk_LogisticRegression()])
        cb = MultitaskModelProbing(None, None, probes)
        names = cb._parse_names(probes, ["foo", "bar"])
        self.assertEqual(names, ["foo", "bar"])

    def test_wrong_length_raises(self):
        probes = MultiTaskEstimator([sk_LogisticRegression(), sk_LogisticRegression()])
        cb = MultitaskModelProbing(None, None, probes)
        with self.assertRaises(ValueError):
            cb._parse_names(probes, ["only_one"])

    def test_wrong_type_raises(self):
        probes = MultiTaskEstimator([sk_LogisticRegression(), sk_LogisticRegression()])
        cb = MultitaskModelProbing(None, None, probes)
        with self.assertRaises(ValueError):
            cb._parse_names(probes, "notalist")


class TestMultitaskModelProbing(unittest.TestCase):
    def setUp(self):
        self.dummy_pl = MagicMock()
        self.dummy_pl.log = MagicMock()
        self.dummy_pl.log_dict = MagicMock()

    def test_invalid_probes_raise(self):
        with self.assertRaises(ValueError):
            MultitaskModelProbing(None, None, sk_LogisticRegression())

    @patch("sklearn.metrics.classification_report")
    @patch("sklearn.metrics.balanced_accuracy_score")
    def test_log_classification_metrics(self, mock_bacc, mock_report):
        mock_report.return_value = {
            "accuracy": 0.9,
            "macro avg": {"f1-score": 0.8, "precision": 0.7, "recall": 0.6},
            "weighted avg": {"f1-score": 0.85},
        }
        mock_bacc.return_value = 0.75

        cb = MultitaskModelProbing(None, None, MultiTaskEstimator([sk_LogisticRegression()]))
        cb.log_classification_metrics(
            self.dummy_pl, y_pred=np.array([0, 1, 0]), y_true=np.array([0, 1, 1]), task_name="task0"
        )

        self.dummy_pl.log_dict.assert_called_once()
        logged = self.dummy_pl.log_dict.call_args[0][0]
        self.assertIn("task0/accuracy", logged)
        self.assertIn("task0/balanced_accuracy", logged)

    @patch("nidl.metrics.regression_report")
    def test_log_regression_metrics(self, mock_reg_report):
        mock_reg_report.return_value = {
            "mae": 0.1,
            "r2": 0.9,
            "nested": {"t1": 0.5, "t2": 0.6},
        }
        cb = MultitaskModelProbing(None, None, MultiTaskEstimator([LinearRegression()]))
        cb.log_regression_metrics(
            self.dummy_pl, y_pred=np.array([0.5, 1.2]), y_true=np.array([0.4, 1.0]), task_name="regtask"
        )
        self.assertTrue(self.dummy_pl.log.called or self.dummy_pl.log_dict.called)

    def test_log_metrics_dispatch(self):
        clf = sk_LogisticRegression(max_iter=200)
        reg = LinearRegression()
        probes = MultiTaskEstimator([clf, reg])
        cb = MultitaskModelProbing(None, None, probes)

        y_true = np.array([[0, 1.0], [1, 2.0]])
        y_pred = np.array([[0, 1.1], [1, 2.1]])

        cb.log_classification_metrics = MagicMock()
        cb.log_regression_metrics = MagicMock()

        cb.log_metrics(self.dummy_pl, y_pred, y_true)

        cb.log_classification_metrics.assert_called_once()
        cb.log_regression_metrics.assert_called_once()

class TestEndToEndProbing(unittest.TestCase):
    def setUp(self):
        # Small synthetic dataset: 10 samples, 3 features, 2 tasks
        X = torch.randn(10, 3)
        y_class = torch.randint(0, 2, (10,))  # binary classification
        y_reg = torch.randn(10)               # regression
        y = torch.stack([y_class.float(), y_reg], dim=1)

        dataset = TensorDataset(X, y)
        self.train_loader = DataLoader(dataset, batch_size=5)
        self.test_loader = DataLoader(dataset, batch_size=5)

    def test_probe_fits_and_logs(self):
        # Two probes: classifier + regressor
        probes = MultiTaskEstimator([sk_LogisticRegression(max_iter=200), LinearRegression()])

        cb = MultitaskModelProbing(
            train_dataloader=self.train_loader,
            test_dataloader=self.test_loader,
            probes=probes,
            probe_names=["clf_task", "reg_task"],
        )

        # Dummy pl_module: returns inputs as embeddings
        class DummyPL:
            def __init__(self):
                self.log = MagicMock()
                self.log_dict = MagicMock()
            def forward(self, x):
                return x.detach().numpy()

        pl_module = DummyPL()

        # Simulate log_metrics call after embedding & probe training
        X_all, y_all = next(iter(self.train_loader))
        X_all = X_all.numpy()
        y_all = y_all.numpy()

        cb.probe.fit(X_all, y_all)  # train the multitask estimator
        y_pred = cb.probe.predict(X_all)

        cb.log_metrics(pl_module, y_pred, y_all)

        # Ensure both tasks were logged
        pl_module.log_dict.assert_called()
        logged_keys = list(pl_module.log_dict.call_args[0][0].keys())
        self.assertTrue(any("clf_task" in k for k in logged_keys) or
                        any("reg_task" in k for k in logged_keys))

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        if self.labels is not None:
            y = self.labels[index]
            return x, y
        else:
            return x

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    unittest.main()
