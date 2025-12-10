import os
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

from nidl.callbacks.metrics import MetricsCallback

try:
    import torchmetrics
    _HAS_TORCHMETRICS = True
except Exception:
    _HAS_TORCHMETRICS = False

NUM_CLASSES = 4
BATCH_SIZE = 8
N_BATCHES = 3
SEED = 1234

def seed_everything(seed=SEED):
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

def make_dataset(device="cpu", n_batches=N_BATCHES, batch_size=BATCH_SIZE, num_classes=NUM_CLASSES):
    """Create a dataset where logits are one-hot aligned with targets (accuracy=1.0)."""
    x = torch.randn(n_batches * batch_size, 10, device=device)
    y = torch.randint(0, num_classes, (n_batches * batch_size,), device=device)
    ds = TensorDataset(x, y)
    return ds

class TinyModule(pl.LightningModule):
    """
    - Returns outputs dict with keys: 'preds', 'targets', 'logits' (and others if needed).
    - Loss is a dummy CE to satisfy PL training requirements.
    """
    def __init__(self, num_classes=NUM_CLASSES, on_gpu=False):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.net = nn.Linear(10, num_classes, bias=False)
        # initialize weights to near zero so the learned logits don't dominate our controlled logits
        nn.init.zeros_(self.net.weight)

    def forward(self, x):
        return self.net(x)

    def _step_common(self, batch, batch_idx):
        x, y = batch
        # Produce controlled logits that match y (one-hot * 10) to make accuracy=1.0
        oh = F.one_hot(y, num_classes=self.num_classes).float()
        logits = oh * 10.0
        preds = logits.argmax(dim=-1)

        # Dummy loss to keep trainer happy (use a tiny forward so graph exists)
        z = self.forward(x.detach())  # random, but tiny since weights=0
        loss = F.cross_entropy( (z + logits.detach()).float(), y)

        # Outputs for metrics callback
        outputs = {
            "preds": preds,      # int64
            "targets": y,        # int64
            "logits": logits,    # float32
            "loss": loss
        }
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self._step_common(batch, batch_idx)
        # return both loss (for optimizer) and outputs (for callback)
        self.log("train_loss", outputs["loss"], on_step=True, prog_bar=False)
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self._step_common(batch, batch_idx)
        self.log("val_loss", outputs["loss"], on_epoch=True, prog_bar=False)
        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self._step_common(batch, batch_idx)
        self.log("test_loss", outputs["loss"], on_epoch=True, prog_bar=False)
        return outputs

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)

def make_loader(device="cpu"):
    ds = make_dataset(device=device)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

def run_trainer(model, train_loader=None, val_loader=None, test_loader=None,
                max_epochs=1, accelerator="cpu", devices=1, strategy="auto",
                callbacks=None, limit_train_batches=N_BATCHES, limit_val_batches=N_BATCHES,
                limit_test_batches=N_BATCHES):
    trainer = pl.Trainer(
        default_root_dir=os.getcwd(),
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=max_epochs,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        callbacks=callbacks or [],
    )
    if train_loader is not None:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    elif val_loader is not None:
        trainer.validate(model, dataloaders=val_loader)
    elif test_loader is not None:
        trainer.test(model, dataloaders=test_loader)
    return trainer

# -----------------------
#        TESTS
# -----------------------

class TestTorchMetricsCPU(unittest.TestCase):
    @unittest.skipUnless(_HAS_TORCHMETRICS, "torchmetrics not installed")
    def test_torchmetrics_accuracy_single_process_cpu(self):
        seed_everything()
        model = TinyModule()
        train_loader = make_loader("cpu")
        # torchmetrics will stream; callback should just call .compute()
        acc = torchmetrics.classification.MulticlassAccuracy(num_classes=NUM_CLASSES)
        metrics = {"acc": acc}

        cb = MetricsCallback(
            metrics=metrics,
            needs=["preds", "targets"],
            compute_per_training_step=False,  # ignored for TM
            every_n_train_steps=None,
            every_n_train_epochs=1,
            every_n_val_epochs=None,
            on_test_end=False,
        )

        trainer = run_trainer(model, train_loader=train_loader, callbacks=[cb])
        # Torchmetrics Accuracy should be 1.0 on our synthetic data
        acc_val = float(trainer.callback_metrics["acc/train"])
        self.assertAlmostEqual(acc_val, 1.0, places=6)

class TestSklearnCPU(unittest.TestCase):
    def test_sklearn_accuracy_per_step_reduced_cpu(self):
        seed_everything()
        model = TinyModule()
        train_loader = make_loader("cpu")

        # sklearn accuracy_score(y_true, y_pred) -> needs positional [targets, preds]
        metrics = {"sk_acc": accuracy_score}

        cb = MetricsCallback(
            metrics=metrics,
            needs=["targets", "preds"],  # positional order for sklearn func
            compute_per_training_step=True,   # compute per step and average (then reduce via strategy)
            every_n_train_steps=1,
            every_n_train_epochs=None,
            every_n_val_epochs=None,
            on_test_end=False,
        )

        trainer = run_trainer(model, train_loader=train_loader, callbacks=[cb])
        acc_val = float(trainer.callback_metrics["sk_acc/train"])
        self.assertAlmostEqual(acc_val, 1.0, places=6)

    def test_sklearn_accuracy_epoch_end_cpu_numpy_path(self):
        seed_everything()
        model = TinyModule()
        train_loader = make_loader("cpu")

        # Wrap sklearn to force a numpy output
        def sk_acc_np(y_true, y_pred):
            return accuracy_score(y_true, y_pred)

        metrics = {"sk_acc_np": sk_acc_np}

        cb = MetricsCallback(
            metrics=metrics,
            needs=["targets", "preds"],
            compute_per_training_step=False,
            compute_per_val_step=False,
            compute_per_test_step=False,
            every_n_train_steps=2, 
            every_n_train_epochs=False,
            on_test_end=False,
        )

        trainer = run_trainer(model, train_loader=train_loader, callbacks=[cb])
        acc_val = float(trainer.callback_metrics["sk_acc_np/train"])
        self.assertAlmostEqual(acc_val, 1.0, places=6)

class TestCustomMetricsCPU(unittest.TestCase):
    def test_custom_metric_kwargs_and_lambda_cpu(self):
        seed_everything()
        model = TinyModule()
        train_loader = make_loader("cpu")

        # custom metric using kwargs and logits->preds transform via lambda in needs
        def acc_from_logits(preds=None, targets=None):
            # preds expected probs/logits; convert to labels
            if isinstance(preds, torch.Tensor):
                y_pred = preds.argmax(dim=-1).cpu().numpy()
                y_true = targets.cpu().numpy()
            elif isinstance(preds, np.ndarray):
                y_pred = preds.argmax(axis=-1)
                y_true = targets
            else:
                raise TypeError("unexpected type")
            return float((y_pred == y_true).mean())

        metrics = {"custom_acc": acc_from_logits}
        needs = {"preds": "logits", "targets": "targets"}  # global kw mapping

        cb = MetricsCallback(
            metrics=metrics,
            needs=needs,
            compute_per_training_step=False,   # cache tensors, compute once
            every_n_train_steps=None,
            every_n_train_epochs=1,
        )

        trainer = run_trainer(model, train_loader=train_loader, callbacks=[cb])
        acc_val = float(trainer.callback_metrics["custom_acc/train"])
        self.assertAlmostEqual(acc_val, 1.0, places=6)

@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestGPU(unittest.TestCase):
    def test_torchmetrics_accuracy_gpu_single_process(self):
        seed_everything()
        model = TinyModule().cuda()
        train_loader = make_loader("cuda")

        if not _HAS_TORCHMETRICS:
            self.skipTest("torchmetrics not installed")

        acc = torchmetrics.classification.MulticlassAccuracy(num_classes=NUM_CLASSES).cuda()
        metrics = {"acc": acc}

        cb = MetricsCallback(
            metrics=metrics,
            needs=["preds", "targets"],
            compute_per_training_step=False,
            every_n_train_epochs=1,
            every_n_train_steps=None,
        )

        trainer = run_trainer(model, train_loader=train_loader, callbacks=[cb],
                              accelerator="gpu", devices=1)
        acc_val = float(trainer.callback_metrics["acc/train"])
        self.assertAlmostEqual(acc_val, 1.0, places=6)

    def test_sklearn_per_step_gpu_scalars(self):
        seed_everything()
        model = TinyModule().cuda()
        train_loader = make_loader("cuda")

        metrics = {"sk_acc": accuracy_score}

        cb = MetricsCallback(
            metrics=metrics,
            needs=["targets", "preds"],
            compute_per_training_step=True,
            compute_on_cpu=True,
            every_n_train_steps=1,
            every_n_train_epochs=None,
        )

        trainer = run_trainer(model, train_loader=train_loader, callbacks=[cb],
                              accelerator="gpu", devices=1)
        acc_val = float(trainer.callback_metrics["sk_acc/train"])
        self.assertAlmostEqual(acc_val, 1.0, places=6)

# ------------- Distributed tests (CPU-DDP) -------------

def _can_run_cpu_ddp():
    # Lightning can run CPU DDP if more than 1 device (process) requested; this will start subprocesses.
    # Many CI environments allow it; if it fails on your CI, guard with env var instead.
    return True

@unittest.skipUnless(_can_run_cpu_ddp(), "CPU DDP not permitted in this environment")
class TestDistributedCPU(unittest.TestCase):
    def test_sklearn_per_step_ddp_cpu(self):
        seed_everything()
        model = TinyModule()
        train_loader = make_loader("cpu")

        metrics = {"sk_acc": accuracy_score}

        cb = MetricsCallback(
            metrics=metrics,
            needs=["targets", "preds"],
            compute_per_training_step=True,
            every_n_train_steps=1,
            every_n_train_epochs=None,
            every_n_val_epochs=None,
        )

        trainer = run_trainer(
            model,
            train_loader=train_loader,
            callbacks=[cb],
            accelerator="cpu",
            devices=2,              # 2 CPU processes via gloo
            strategy="ddp",         # Lightning will use gloo on CPU
        )
        # Reduced value should still be 1.0 on rank zero
        if trainer.is_global_zero:
            acc_val = float(trainer.callback_metrics["sk_acc/train"])
            self.assertAlmostEqual(acc_val, 1.0, places=6)

    @unittest.skipUnless(_HAS_TORCHMETRICS, "torchmetrics not installed")
    def test_torchmetrics_epoch_end_ddp_cpu(self):
        seed_everything()
        model = TinyModule()
        train_loader = make_loader("cpu")

        acc = torchmetrics.classification.MulticlassAccuracy(num_classes=NUM_CLASSES)
        metrics = {"acc": acc}

        cb = MetricsCallback(
            metrics=metrics,
            needs=["preds", "targets"],
            compute_per_training_step=False,   # ignored for TM
            every_n_train_steps=None,
            every_n_train_epochs=1,
        )

        trainer = run_trainer(
            model,
            train_loader=train_loader,
            callbacks=[cb],
            accelerator="cpu",
            devices=2,
            strategy="ddp",
        )
        if trainer.is_global_zero:
            acc_val = float(trainer.callback_metrics["acc/train"])
            self.assertAlmostEqual(acc_val, 1.0, places=6)

# ------------- Distributed tests (CUDA-DDP) -------------

def _can_run_cuda_ddp():
    return torch.cuda.is_available() and torch.cuda.device_count() >= 2

@unittest.skipUnless(_can_run_cuda_ddp(), "Need >=2 CUDA GPUs for CUDA DDP tests")
class TestDistributedCUDA(unittest.TestCase):
    def test_sklearn_per_step_ddp_cuda(self):
        seed_everything()
        model = TinyModule().cuda()
        train_loader = make_loader("cuda")

        metrics = {"sk_acc": accuracy_score}

        cb = MetricsCallback(
            metrics=metrics,
            needs=["targets", "preds"],
            compute_per_training_step=True,
            compute_on_cpu=True,
            every_n_train_steps=1,
            every_n_train_epochs=None,
        )

        trainer = run_trainer(
            model,
            train_loader=train_loader,
            callbacks=[cb],
            accelerator="gpu",
            devices=2,
            strategy="ddp",
        )
        if trainer.is_global_zero:
            acc_val = float(trainer.callback_metrics["sk_acc/train"])
            self.assertAlmostEqual(acc_val, 1.0, places=6)

    @unittest.skipUnless(_HAS_TORCHMETRICS, "torchmetrics not installed")
    def test_torchmetrics_epoch_end_ddp_cuda(self):
        seed_everything()
        model = TinyModule().cuda()
        train_loader = make_loader("cuda")

        acc = torchmetrics.classification.MulticlassAccuracy(num_classes=NUM_CLASSES).cuda()
        metrics = {"acc": acc}

        cb = MetricsCallback(
            metrics=metrics,
            needs=["preds", "targets"],
            compute_per_training_step=False,
            every_n_train_steps=None,
            every_n_train_epochs=1,
        )

        trainer = run_trainer(
            model,
            train_loader=train_loader,
            callbacks=[cb],
            accelerator="gpu",
            devices=2,
            strategy="ddp",
        )
        if trainer.is_global_zero:
            acc_val = float(trainer.callback_metrics["acc/train"])
            self.assertAlmostEqual(acc_val, 1.0, places=6)

# ------------- Edge cases -------------

class TestEdgeCases(unittest.TestCase):
    def test_needs_per_metric_overrides(self):
        seed_everything()
        model = TinyModule()
        train_loader = make_loader("cpu")

        # Define two metrics with different needs signatures
        def acc_labels_first(y_true, y_pred):
            return float((np.array(y_true) == np.array(y_pred)).mean())

        def acc_logits_kw(preds=None, targets=None):
            p = preds if isinstance(preds, np.ndarray) else preds.cpu().numpy()
            t = targets if isinstance(targets, np.ndarray) else targets.cpu().numpy()
            return float((p.argmax(axis=-1) == t).mean())

        metrics = {"m1": acc_labels_first, "m2": acc_logits_kw}
        needs = {
            "m1": ["targets", "preds"],              # positional
            "m2": {"preds": "logits", "targets": "targets"},  # keyword
        }

        cb = MetricsCallback(
            metrics=metrics,
            needs=needs,
            compute_per_training_step=False,
            every_n_train_steps=None,
            every_n_train_epochs=1,
        )

        trainer = run_trainer(model, train_loader=train_loader, callbacks=[cb])
        self.assertAlmostEqual(float(trainer.callback_metrics["m1/train"]), 1.0, places=6)
        self.assertAlmostEqual(float(trainer.callback_metrics["m2/train"]), 1.0, places=6)

    def test_numpy_scalar_and_python_scalar_outputs(self):
        seed_everything()
        model = TinyModule()
        train_loader = make_loader("cpu")

        def numpy_scalar_metric(preds, targets):
            return np.array(1.0)

        def python_scalar_metric(preds, targets):
            return 1.0

        metrics = {"np_scalar": numpy_scalar_metric, "py_scalar": python_scalar_metric}

        cb = MetricsCallback(
            metrics=metrics,
            needs=["preds", "targets"],
            compute_per_training_step=True,
            every_n_train_steps=1,
        )

        trainer = run_trainer(model, train_loader=train_loader, callbacks=[cb])
        self.assertAlmostEqual(float(trainer.callback_metrics["np_scalar/train"]), 1.0, places=6)
        self.assertAlmostEqual(float(trainer.callback_metrics["py_scalar/train"]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
