# tests/test_model_probing_cv_callback.py

import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import socket
import os
import pytorch_lightning as pl
from sklearn.linear_model import LogisticRegression

from nidl.estimators.base import BaseEstimator, TransformerMixin
from nidl.callbacks.model_probing_cv import ModelProbingCV


def make_linearly_separable_dataset(n_per_class: int = 16, dim: int = 2):
    """Simple, perfectly separable dataset for classification."""
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
        s.bind(("", 0))  # OS chooses a free port
        return s.getsockname()[1]

class DummyEmbeddingModule(TransformerMixin, BaseEstimator):
    """LightningModule exposing transform_step as identity embedding."""

    def __init__(self, input_dim: int = 2):
        super().__init__()
        self.layer = torch.nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.layer(x)

    def transform_step(self, x, batch_idx=None):
        # Identity features: exactly the original X used in the DataLoader
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def test_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class TestModelProbingCVIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        dataset = make_linearly_separable_dataset(n_per_class=10, dim=2)
        cls.dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        cls.test_dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    def _build_callback(
        self,
        scoring="accuracy",
        cv=3,
        n_jobs=None,
        every_n_train_epochs=1,
        every_n_val_epochs=None,
        on_test_epoch_start=False,
        on_test_epoch_end=False,
        prog_bar=False,
        prefix_score="",
    ):
        probe = LogisticRegression(solver="liblinear")
        cb = ModelProbingCV(
            dataloader=self.dataloader,
            probe=probe,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            every_n_train_epochs=every_n_train_epochs,
            every_n_val_epochs=every_n_val_epochs,
            on_test_epoch_start=on_test_epoch_start,
            on_test_epoch_end=on_test_epoch_end,
            prog_bar=prog_bar,
            prefix_score=prefix_score,
        )
        return cb

    def _build_trainer(self, accelerator="cpu", devices=1, strategy="auto", callbacks=None):
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            limit_train_batches=2,
            limit_val_batches=0,
            callbacks=callbacks,
        )
        return trainer

    # ------------------------------------------------------------------
    # Single CPU, end-of-train probing
    # ------------------------------------------------------------------
    def test_single_cpu_train_epoch_end_logs_per_fold_test_score(self):
        callback = self._build_callback(
            scoring="accuracy",
            cv=3,
            every_n_train_epochs=1,
            prefix_score="",  # no prefix
        )
        model = DummyEmbeddingModule(input_dim=2)
        trainer = self._build_trainer(accelerator="cpu", devices=1, callbacks=[callback])

        trainer.fit(model, train_dataloaders=self.dataloader)

        # cross_validate with scoring="accuracy" yields "test_score"
        # ModelProbingCV.log_metrics logs: f"fold{i}/{prefix}{key}"
        # Here key == "test_score".
        for i in range(3):
            key = f"fold{i}/test_score"
            self.assertIn(key, trainer.callback_metrics)
            acc = trainer.callback_metrics[key].item()
            self.assertGreaterEqual(acc, 0.9)

    # ------------------------------------------------------------------
    # Single CPU, test epoch end + prefix
    # ------------------------------------------------------------------
    def test_single_cpu_test_epoch_end_with_prefix(self):
        callback = self._build_callback(
            scoring="accuracy",
            cv=4,
            every_n_train_epochs=None,      # disabled on train
            on_test_epoch_end=True,
            prefix_score="logreg_",
        )
        model = DummyEmbeddingModule(input_dim=2)
        trainer = self._build_trainer(accelerator="cpu", devices=1, callbacks=[callback])

        trainer.fit(model, train_dataloaders=self.dataloader)
        trainer.test(model, dataloaders=self.dataloader)

        for i in range(4):
            key = f"fold{i}/logreg_test_score"
            self.assertIn(key, trainer.callback_metrics)
            acc = trainer.callback_metrics[key].item()
            self.assertGreaterEqual(acc, 0.9)

    # ------------------------------------------------------------------
    # Multi-metric scoring (dict/list) – still using real cross_validate
    # ------------------------------------------------------------------
    def test_multi_metric_scoring_logs_multiple_keys_per_fold(self):
        # scoring dict: names => metrics, so cross_validate keys will be:
        # "test_acc", "test_f1"
        scoring = {"acc": "accuracy", "f1": "f1_macro"}
        callback = self._build_callback(
            scoring=scoring,
            cv=3,
            prefix_score="lin_",
        )
        model = DummyEmbeddingModule(input_dim=2)
        trainer = self._build_trainer(accelerator="cpu", devices=1, callbacks=[callback])

        trainer.fit(model, train_dataloaders=self.dataloader)

        for i in range(3):
            k_acc = f"fold{i}/lin_test_acc"
            k_f1 = f"fold{i}/lin_test_f1"
            self.assertIn(k_acc, trainer.callback_metrics)
            self.assertIn(k_f1, trainer.callback_metrics)
            self.assertGreaterEqual(trainer.callback_metrics[k_acc].item(), 0.8)
            self.assertGreaterEqual(trainer.callback_metrics[k_f1].item(), 0.8)

    # ------------------------------------------------------------------
    # DDP-like test on CPU (ddp_spawn)
    # ------------------------------------------------------------------
    @unittest.skipIf(
        not torch.distributed.is_available(),
        "torch.distributed not available; skipping DDP test.",
    )
    def test_ddp_spawn_cpu_two_devices_logs_fold_metrics(self):
        # dynamically choose a free port for THIS test
        port = find_free_port()
        os.environ["MASTER_PORT"] = str(port)
        
        callback = self._build_callback(
            scoring="accuracy",
            cv=3,
            every_n_train_epochs=1,
        )
        model = DummyEmbeddingModule(input_dim=2)

        trainer = self._build_trainer(
            accelerator="cpu",
            devices=2,
            strategy="ddp_spawn",
            callbacks=[callback],
        )

        trainer.fit(model, train_dataloaders=self.dataloader)

        for i in range(3):
            key = f"fold{i}/test_score"
            self.assertIn(key, trainer.callback_metrics)
            acc = trainer.callback_metrics[key].item()
            self.assertGreaterEqual(acc, 0.8)

    # ------------------------------------------------------------------
    # adapt_dataloader_for_ddp static method
    # ------------------------------------------------------------------
    def test_adapt_dataloader_for_ddp_single_process_returns_same(self):
        trainer = type("T", (), {"world_size": 1, "global_rank": 0})()
        adapted = ModelProbingCV.adapt_dataloader_for_ddp(self.dataloader, trainer)
        self.assertIs(adapted, self.dataloader)

    def test_adapt_dataloader_for_ddp_multi_process_uses_distributed_sampler(self):
        from torch.utils.data import DistributedSampler

        trainer = type("T", (), {"world_size": 2, "global_rank": 1})()
        dl = DataLoader(
            self.dataloader.dataset,
            batch_size=4,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        adapted = ModelProbingCV.adapt_dataloader_for_ddp(dl, trainer)
        self.assertIsInstance(adapted.sampler, DistributedSampler)
        self.assertEqual(adapted.sampler.num_replicas, trainer.world_size)
        self.assertEqual(adapted.sampler.rank, trainer.global_rank)
        self.assertEqual(adapted.batch_size, dl.batch_size)
        self.assertEqual(adapted.num_workers, dl.num_workers)
        self.assertEqual(adapted.pin_memory, dl.pin_memory)

    # ------------------------------------------------------------------
    # cross_validate wrapper (unit-ish)
    # ------------------------------------------------------------------
    def test_cross_validate_passes_parameters_to_sklearn(self):
        from nidl.callbacks import model_probing_cv as mp_mod  # import module

        callback = self._build_callback(
            scoring="accuracy",
            cv=5,
            n_jobs=2,
        )

        X = np.random.randn(20, 2)
        y = np.random.randint(0, 2, size=20)

        with patch.object(mp_mod, "cross_validate") as mock_cv:
            mock_cv.return_value = {"test_score": np.array([1.0])}
            scores = callback.cross_validate(X, y)

        mock_cv.assert_called_once()
        args, kwargs = mock_cv.call_args
        self.assertIs(args[0], callback.probe)
        np.testing.assert_allclose(args[1], X)
        np.testing.assert_allclose(args[2], y)
        self.assertEqual(kwargs["scoring"], callback.scoring)
        self.assertEqual(kwargs["cv"], callback.cv)
        self.assertEqual(kwargs["n_jobs"], callback.n_jobs)
        self.assertFalse(kwargs["return_train_score"])
        self.assertFalse(kwargs["return_estimator"])
        self.assertFalse(kwargs["return_indices"])
        self.assertEqual(scores, mock_cv.return_value)

    # ------------------------------------------------------------------
    # check_array helper
    # ------------------------------------------------------------------
    def test_check_array_enforces_2d_X_and_1d_y(self):
        callback = self._build_callback()
        X = np.random.randn(10, 2)
        y = np.random.randint(0, 2, size=10)

        X_checked, y_checked = callback.check_array(X, y)
        self.assertEqual(X_checked.shape, (10, 2))
        self.assertEqual(y_checked.shape, (10,))

    # ------------------------------------------------------------------
    # probes error paths in probing()
    # ------------------------------------------------------------------
    def test_probing_raises_if_not_base_estimator(self):
        # temporarily patch BaseEstimator in this test only
        with patch(
            "nidl.callbacks.model_probing_cv.BaseEstimator", new=pl.LightningModule
        ), patch(
            "nidl.callbacks.model_probing_cv._estimator_is", return_value=True
        ):
            callback = self._build_callback()
            trainer = self._build_trainer()

            # Use a plain object that is NOT a LightningModule
            pl_module = object()
            with self.assertRaises(ValueError):
                callback.probing(trainer, pl_module)

    def test_probing_raises_if_not_transformer(self):
        # BaseEstimator is LightningModule, but _estimator_is returns False
        with patch(
            "nidl.callbacks.model_probing_cv._estimator_is", return_value=False
        ):
            callback = self._build_callback()
            trainer = self._build_trainer()
            pl_module = DummyEmbeddingModule(input_dim=2)
            with self.assertRaises(ValueError):
                callback.probing(trainer, pl_module)

    # ------------------------------------------------------------------
    # extract_features low-level behavior
    # ------------------------------------------------------------------
    def test_extract_features_single_cpu(self):
        callback = self._build_callback()
        model = DummyEmbeddingModule(input_dim=2)
        trainer = self._build_trainer(devices=1, accelerator="cpu")

        trainer.fit(model, train_dataloaders=self.dataloader)

        X, y = callback.extract_features(trainer, model, self.test_dataloader)

        self.assertEqual(X.shape[0], len(self.test_dataloader.dataset))
        self.assertEqual(y.shape[0], len(self.test_dataloader.dataset))

        original = torch.cat([x for (x, _) in self.test_dataloader], dim=0).numpy()
        np.testing.assert_allclose(X, original, rtol=1e-6, atol=1e-6)

    # ------------------------------------------------------------------
    # hook behavior (on_train_epoch_end / on_validation_epoch_end)
    # ------------------------------------------------------------------
    def test_on_train_epoch_end_respects_every_n_train_epochs(self):
        callback = self._build_callback(every_n_train_epochs=2)
        model = DummyEmbeddingModule(input_dim=2)

        with patch.object(callback, "probing") as mock_probing:
            trainer = type(
                "T", (), {"current_epoch": 0, "is_global_zero": True}
            )()

            # epoch 0 -> 0 % 2 == 0 -> probing called
            callback.on_train_epoch_end(trainer, model)
            mock_probing.assert_called_once_with(trainer, model)

            # epoch 1 -> 1 % 2 != 0 -> no probing
            mock_probing.reset_mock()
            trainer.current_epoch = 1
            callback.on_train_epoch_end(trainer, model)
            mock_probing.assert_not_called()

    def test_on_validation_epoch_end_uses_counter_and_every_n_val_epochs(self):
        callback = self._build_callback(every_n_val_epochs=2)
        model = DummyEmbeddingModule(input_dim=2)
        trainer = type("T", (), {"is_global_zero": True})()

        with patch.object(callback, "probing") as mock_probing:
            # counter_val_epochs starts at 0, so 0 % 2 == 0 => probing
            callback.on_validation_epoch_end(trainer, model)
            mock_probing.assert_called_once_with(trainer, model)
            self.assertEqual(callback.counter_val_epochs, 1)

            # counter_val_epochs == 1, 1 % 2 != 0 => no probing
            mock_probing.reset_mock()
            callback.on_validation_epoch_end(trainer, model)
            self.assertEqual(callback.counter_val_epochs, 2)
            mock_probing.assert_not_called()

    def test_on_test_epoch_start_and_end_flags(self):
        callback = self._build_callback(
            every_n_train_epochs=None,
            every_n_val_epochs=None,
            on_test_epoch_start=True,
            on_test_epoch_end=False,
        )
        model = DummyEmbeddingModule(input_dim=2)
        trainer = type("T", (), {"is_global_zero": True})()

        with patch.object(callback, "probing") as mock_probing:
            callback.on_test_epoch_start(trainer, model)
            mock_probing.assert_called_once_with(trainer, model)

            mock_probing.reset_mock()
            callback.on_test_epoch_end(trainer, model)
            mock_probing.assert_not_called()

        # Flip flags
        callback._on_test_epoch_start = False
        callback._on_test_epoch_end = True
        with patch.object(callback, "probing") as mock_probing:
            callback.on_test_epoch_start(trainer, model)
            mock_probing.assert_not_called()

            callback.on_test_epoch_end(trainer, model)
            mock_probing.assert_called_once_with(trainer, model)


if __name__ == "__main__":
    unittest.main()
