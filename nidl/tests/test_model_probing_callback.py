import unittest
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import socket
import os
import pytorch_lightning as pl
from sklearn.linear_model import LogisticRegression

from nidl.estimators import BaseEstimator, TransformerMixin
from nidl.callbacks.model_probing import ModelProbing

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
        s.bind(("", 0))  # OS chooses a free port
        return s.getsockname()[1]

class DummyEmbeddingModule(TransformerMixin, BaseEstimator):
    """Simple LightningModule that exposes transform_step.

    - transform_step returns the raw input features (identity),
      so the probe sees exactly the original dataset.
    """

    def __init__(self, dim=2):
        super().__init__()
        self.linear = torch.nn.Linear(dim, 2)

    def forward(self, x):
        return self.linear(x)

    def transform_step(self, x, batch_idx=None):
        # Identity embedding: features are just the inputs
        return x
    
    def test_step(self, batch, batch_idx):
        return None

    def training_step(self, batch, batch_idx): # keep PL busy
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class TestModelProbingIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Small deterministic dataset
        torch.manual_seed(0)
        dataset = make_linearly_separable_dataset(n_per_class=10, dim=2)
        cls.train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        cls.test_loader = DataLoader(dataset, batch_size=8, shuffle=False)

    def _build_probing_callback(
        self,
        scoring="accuracy",
        every_n_train_epochs=1,
        every_n_val_epochs=None,
        on_test_epoch_start=False,
        on_test_epoch_end=False,
        prefix_score="",
        prog_bar=False,
    ):
        probe = LogisticRegression(solver="liblinear")
        cb = ModelProbing(
            train_dataloader=self.train_loader,
            test_dataloader=self.test_loader,
            probe=probe,
            scoring=scoring,
            every_n_train_epochs=every_n_train_epochs,
            every_n_val_epochs=every_n_val_epochs,
            on_test_epoch_start=on_test_epoch_start,
            on_test_epoch_end=on_test_epoch_end,
            prog_bar=prog_bar,
            prefix_score=prefix_score,
        )
        return cb

    def _build_trainer(
        self,
        accelerator="cpu",
        devices=1,
        strategy="auto",
        callbacks=None,
    ):
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            limit_train_batches=2,  # keep tests fast
            limit_val_batches=0,
            callbacks=callbacks
        )
        return trainer

    def test_single_cpu_train_epoch_end_logs_accuracy(self):
        """End-of-train probing on single CPU should log a numeric 'test_score'."""
        callback = self._build_probing_callback(
            scoring="accuracy",
            every_n_train_epochs=1,
            every_n_val_epochs=None,
            prefix_score="",  # default
        )
        model = DummyEmbeddingModule()
        trainer = self._build_trainer(
            accelerator="cpu", devices=1, callbacks=[callback]
        )

        trainer.fit(model, train_dataloaders=self.train_loader)

        # ModelProbing.log_metrics uses "<prefix>test_score" for single score
        self.assertIn("test_score", trainer.callback_metrics)
        acc = trainer.callback_metrics["test_score"].item()
        # On a perfectly separable dataset, logistic regression should reach 1.0
        self.assertGreaterEqual(acc, 0.95)

    def test_single_cpu_test_epoch_end_logs_prefixed_accuracy(self):
        """Probing at test epoch end, with prefix, logs '<prefix>test_score'."""
        callback = self._build_probing_callback(
            scoring="accuracy",
            every_n_train_epochs=None,
            on_test_epoch_end=True,
            prefix_score="logreg_",
        )
        model = DummyEmbeddingModule()
        trainer = self._build_trainer(
            accelerator="cpu", devices=1, callbacks=[callback]
        )

        # We still run one tiny training epoch to be closer to real usage
        trainer.fit(model, train_dataloaders=self.train_loader)
        trainer.test(model, dataloaders=self.test_loader)

        self.assertIn("logreg_test_score", trainer.callback_metrics)
        acc = trainer.callback_metrics["logreg_test_score"].item()
        self.assertGreaterEqual(acc, 0.95)

    @unittest.skipIf(
        not torch.distributed.is_available(),
        "Torch distributed not available; skipping DDP test.",
    )
    def test_ddp_spawn_cpu_two_devices_logs_accuracy(self):
        """Run with a DDP-like setup on CPU (2 processes) and ensure probing works.

        This test uses the real Trainer with strategy='ddp_spawn' and 2 devices.
        It exercises:
        - adapt_dataloader_for_ddP (DistributedSampler)
        - extract_features gathering & flattening across ranks
        - log_metrics & aggregation to rank 0
        """

        # dynamically choose a free port for THIS test
        port = find_free_port()
        os.environ["MASTER_PORT"] = str(port)

        callback = self._build_probing_callback(
            scoring="accuracy",
            every_n_train_epochs=1,
        )
        model = DummyEmbeddingModule()

        trainer = self._build_trainer(
            accelerator="cpu",
            devices=2,
            strategy="ddp_spawn",
            callbacks=[callback],
        )

        trainer.fit(model, train_dataloaders=self.train_loader)

        # After fit returns, callback_metrics is available on the main process.
        self.assertIn("test_score", trainer.callback_metrics)
        acc = trainer.callback_metrics["test_score"].item()
        self.assertGreaterEqual(acc, 0.95)

    def test_log_metrics_dict_scores_path(self):
        """Exercise the dict-scoring branch of log_metrics.
        """
        callback = self._build_probing_callback(
            scoring={"accuracy": lambda est, X, y: 0.7,
                     "f1": lambda est, X, y: 0.5},
            prefix_score="custom_",
            prog_bar=True,
        )
        model = DummyEmbeddingModule()
        trainer = self._build_trainer(
            accelerator="cpu", devices=1, callbacks=[callback]
        )
        trainer.fit(model, train_dataloaders=self.train_loader)

        self.assertIn("custom_test_accuracy", trainer.callback_metrics)
        self.assertIn("custom_test_f1", trainer.callback_metrics)
        self.assertAlmostEqual(
            trainer.callback_metrics["custom_test_accuracy"].item(), 0.7
        )
        self.assertAlmostEqual(
            trainer.callback_metrics["custom_test_f1"].item(), 0.5
        )


if __name__ == "__main__":
    unittest.main()
