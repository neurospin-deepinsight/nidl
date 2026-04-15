import os
import socket
import unittest

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader, TensorDataset

from nidl.estimators.dummy import DummyEmbeddingEstimator
from nidl.estimators.probes import ModelProbing


def make_linearly_separable_dataset(n_per_class: int = 16, dim: int = 2):
    """Simple perfectly linearly separable dataset.

    Class 0: all features = -1
    Class 1: all features = +1
    """
    x0 = -torch.ones(n_per_class, dim)
    x1 = torch.ones(n_per_class, dim)
    X = torch.cat([x0, x1], dim=0)
    y = torch.cat(
        [
            torch.zeros(n_per_class, dtype=torch.long),
            torch.ones(n_per_class, dtype=torch.long),
        ],
        dim=0,
    )
    return TensorDataset(X, y)


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TestModelProbingIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

        dataset = make_linearly_separable_dataset(n_per_class=10, dim=2)
        cls.train_loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
        )
        cls.test_loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
        )
        cls.X_true = dataset.tensors[0]
        cls.y_true = dataset.tensors[1].numpy()

    def _build_probing(
        self,
        scoring="accuracy",
        dummy_strategy="normal",
        accelerator="cpu",
        devices=1,
        strategy="auto",
        callbacks=None,
    ):
        probe = LogisticRegression(solver="liblinear", max_iter=200, random_state=0)
        return ModelProbing(
            embedding_estimator=DummyEmbeddingEstimator(strategy=dummy_strategy),
            probe=probe,
            scoring=scoring,
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            callbacks=callbacks,
        )

    def test_init_rejects_invalid_probe(self):
        with self.assertRaises(TypeError):
            ModelProbing(
                embedding_estimator=DummyEmbeddingEstimator(strategy="normal"),
                probe="not_a_sklearn_estimator",
                accelerator="cpu",
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_model_summary=False,
            )
    
    def test_init_rejects_invalid_embedding_estimator(self):
        with self.assertRaises(TypeError):
            ModelProbing(
                embedding_estimator="not_an_estimator",
                probe=LogisticRegression(),
                accelerator="cpu",
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_model_summary=False,
            )

    def test_fit_accepts_and_ignores_val_dataloader(self):
        model = self._build_probing()
        returned = model.fit(
            train_dataloader=self.train_loader,
            val_dataloader=self.test_loader,
        )
        self.assertIs(returned, model)
        self.assertTrue(model.fitted_)

    def test_score_uses_probe_default_scoring_when_none(self):
        model = self._build_probing(scoring=None)
        model.fit(self.train_loader)
        score = model.score(self.test_loader)
        self.assertIsInstance(score, float)

    def test_score_with_dict_scoring(self):
        model = self._build_probing(
            scoring={
                "acc": "accuracy",
                "f1": "f1",
            }
        )
        model.fit(self.train_loader)
        metrics = model.score(self.test_loader)
        self.assertIn("acc", metrics)
        self.assertIn("f1", metrics)
    
    def test_score_with_callable_scoring(self):
        def scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            return accuracy_score(y, y_pred)

        model = self._build_probing(scoring=scorer, dummy_strategy="identity")
        model.fit(self.train_loader)
        score = model.score(self.test_loader)
        self.assertAlmostEqual(score, 1.0)

    def test_predict_matches_expected_labels(self):
        model = self._build_probing(dummy_strategy="identity")
        model.fit(self.train_loader)
        y_pred = model.predict(self.test_loader)
        self.assertTrue((y_pred == self.y_true).all())

    def test_predict_before_fit_raises(self):
        model = self._build_probing()
        with self.assertRaises(Exception):
            model.predict(test_dataloader=self.test_loader)

    def test_score_before_fit_raises(self):
        model = self._build_probing()
        with self.assertRaises(Exception):
            model.score(test_dataloader=self.test_loader)

    def test_fit_sets_fitted_flag_and_probe(self):
        model = self._build_probing(scoring="accuracy", dummy_strategy="normal")

        returned = model.fit(train_dataloader=self.train_loader)

        self.assertIs(returned, model)
        self.assertTrue(getattr(model, "fitted_", False))
        check_is_fitted(model.probe)

    def test_single_cpu_score_matches_sklearn_accuracy(self):
        model = self._build_probing(
            scoring="accuracy",
            dummy_strategy="normal",
            accelerator="cpu",
            devices=1,
        )
        model.fit(train_dataloader=self.train_loader)

        acc = model.score(test_dataloader=self.test_loader)
        y_pred = model.predict(test_dataloader=self.test_loader)

        self.assertEqual(len(y_pred), len(self.y_true))
        self.assertAlmostEqual(acc, accuracy_score(self.y_true, y_pred))

    def test_single_cpu_multiscores(self):
        model = self._build_probing(
            scoring=["accuracy", "f1"],
            dummy_strategy="normal",
            accelerator="cpu",
            devices=1,
        )
        model.fit(train_dataloader=self.train_loader)

        metrics = model.score(test_dataloader=self.test_loader)
        y_pred = model.predict(test_dataloader=self.test_loader)

        self.assertIsInstance(metrics, dict)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1", metrics)
        self.assertAlmostEqual(metrics["accuracy"], accuracy_score(self.y_true, y_pred))
        self.assertAlmostEqual(metrics["f1"], f1_score(self.y_true, y_pred))

    def test_score_override_at_call_time(self):
        model = self._build_probing(
            scoring="accuracy",
            dummy_strategy="normal",
            accelerator="cpu",
            devices=1,
        )
        model.fit(train_dataloader=self.train_loader)

        acc = model.score(test_dataloader=self.test_loader)
        f1 = model.score(test_dataloader=self.test_loader, scoring="f1")

        y_pred = model.predict(test_dataloader=self.test_loader)

        self.assertAlmostEqual(acc, accuracy_score(self.y_true, y_pred))
        self.assertAlmostEqual(f1, f1_score(self.y_true, y_pred))

    def test_predict_returns_expected_shape(self):
        model = self._build_probing(
            scoring="accuracy",
            dummy_strategy="identity",
            accelerator="cpu",
            devices=1,
        )
        model.fit(train_dataloader=self.train_loader)

        y_pred = model.predict(test_dataloader=self.test_loader)

        self.assertEqual(y_pred.shape[0], len(self.y_true))

    @unittest.skipIf(
        not torch.distributed.is_available(),
        "Torch distributed not available; skipping DDP test.",
    )
    def test_ddp_spawn_cpu_two_devices(self):
        port = find_free_port()
        old_master_port = os.environ.get("MASTER_PORT")
        os.environ["MASTER_PORT"] = str(port)

        try:
            model = self._build_probing(
                scoring="accuracy",
                dummy_strategy="identity",
                accelerator="cpu",
                devices=2,
                strategy="ddp_spawn",
            )
            model.fit(train_dataloader=self.train_loader)
            acc = model.score(test_dataloader=self.test_loader)
            self.assertGreaterEqual(acc, 0.95)
        finally:
            if old_master_port is None:
                os.environ.pop("MASTER_PORT", None)
            else:
                os.environ["MASTER_PORT"] = old_master_port

    @unittest.skipIf(
        not torch.distributed.is_available(),
        "Torch distributed not available; skipping DDP comparison test.",
    )
    def test_ddp_spawn_cpu_two_devices_vs_one_device(self):
        port = find_free_port()
        old_master_port = os.environ.get("MASTER_PORT")
        os.environ["MASTER_PORT"] = str(port)

        try:
            ddp_model = self._build_probing(
                scoring="accuracy",
                dummy_strategy="identity",
                accelerator="cpu",
                devices=2,
                strategy="ddp_spawn",
            )
            single_model = self._build_probing(
                scoring="accuracy",
                dummy_strategy="identity",
                accelerator="cpu",
                devices=1,
                strategy="auto",
            )

            ddp_model.fit(train_dataloader=self.train_loader)
            single_model.fit(train_dataloader=self.train_loader)

            ddp_acc = ddp_model.score(test_dataloader=self.test_loader)
            single_acc = single_model.score(test_dataloader=self.test_loader)

            self.assertAlmostEqual(ddp_acc, single_acc, places=6)
        finally:
            if old_master_port is None:
                os.environ.pop("MASTER_PORT", None)
            else:
                os.environ["MASTER_PORT"] = old_master_port


if __name__ == "__main__":
    unittest.main()