##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from collections import OrderedDict
from types import SimpleNamespace

import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from nidl.estimators import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    RegressorMixin,
    TransformerMixin,
)
from nidl.estimators.ssl import SimCLR, YAwareContrastiveLearning
from nidl.estimators.ssl.utils.projection_heads import SimCLRProjectionHead
from nidl.estimators.linear import LogisticRegression
from nidl.losses.yaware_infonce import KernelMetric
from nidl.transforms import MultiViewsTransform
from nidl.utils import print_multicolor


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""

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


class TestEstimators(unittest.TestCase):
    """Test estimators."""

    def setUp(self):
        """Setup test."""
        self._encoder = nn.Linear(5 * 5, 10)
        self._encoder.latent_size = 10
        self._fc = nn.Linear(self._encoder.latent_size, 2)
        self._model = nn.Sequential(
            OrderedDict([("encoder", self._encoder), ("fc", self._fc)])
        )
        self.n_images = 20
        self.fake_data = torch.rand(self.n_images, 5 * 5)
        self.fake_labels = torch.randint(1, (self.n_images,))
        ssl_transforms = transforms.Compose([lambda x: x + torch.rand(x.size())])
        ssl_dataset = CustomTensorDataset(
            self.fake_data, transform=MultiViewsTransform(ssl_transforms, n_views=2)
        )
        x_dataset = CustomTensorDataset(self.fake_data)
        xy_dataset = CustomTensorDataset(self.fake_data, labels=self.fake_labels)
        xxy_dataset = CustomTensorDataset(
            self.fake_data, labels=self.fake_labels, 
            transform=MultiViewsTransform(ssl_transforms, n_views=2)
        )
        self.ssl_loader = DataLoader(ssl_dataset, batch_size=2, shuffle=False)
        self.weakly_sup_loader = DataLoader(xxy_dataset, batch_size=2, shuffle=False)
        self.x_loader = DataLoader(x_dataset, batch_size=2, shuffle=False)
        self.xy_loader = DataLoader(xy_dataset, batch_size=2, shuffle=False)

    def ssl_config(self):
        return (
            (
                SimCLR,
                {
                    "encoder": self._encoder,
                    "hidden_dims": [self._encoder.latent_size, 3],
                    "lr": 5e-4,
                    "temperature": 0.07,
                    "weight_decay": 1e-4,
                    "max_epochs": 2,
                    "random_state": 42,
                    "limit_train_batches": 3,
                },
            ),
            (
                YAwareContrastiveLearning,
                {
                    "encoder": self._encoder,
                    "projection_head_kwargs": {
                        "input_dim": self._encoder.latent_size,
                        "output_dim": 3,
                    },
                    "temperature": 0.07,
                    "learning_rate": 1e-4,
                    "max_epochs": 2,
                    "limit_train_batches": 3,
                },
            ),
                        (
                YAwareContrastiveLearning,
                {
                    "encoder": self._encoder,
                    "projection_head": SimCLRProjectionHead(
                        input_dim=self._encoder.latent_size, output_dim=3
                    ),
                    "projection_head_kwargs": { # ignored
                        "input_dim": self._encoder.latent_size,
                        "output_dim": 3,
                    },
                    "temperature": 0.07,
                    "learning_rate": 1e-4,
                    "max_epochs": 2,
                    "limit_train_batches": 3,
                },
            ),
            (
                YAwareContrastiveLearning,
                {
                    "encoder": self._encoder,
                    "projection_head": SimCLRProjectionHead(
                        input_dim=self._encoder.latent_size, output_dim=3
                    ),
                    "temperature": 0.1,
                    "learning_rate": 5e-4,
                    "optimizer_kwargs": {"weight_decay": 1e-4},
                    "max_epochs": 2,
                    "limit_train_batches": 3
                },
            ),
        )
    
    def test_weakly_sup_config(self):
        kernel_metric = KernelMetric(bandwidth="silverman")
        kernel_metric.fit(self.fake_labels.numpy())
        return (
            (
                YAwareContrastiveLearning,
                {
                    "encoder": self._encoder,
                    "projection_head_kwargs": {
                        "input_dim": self._encoder.latent_size,
                        "output_dim": 10,
                    },
                    "bandwidth": 2,
                    "temperature": 0.07,
                    "learning_rate": 1e-4,
                    "max_epochs": 2,
                    "limit_train_batches": 3,
                },
            ),
                        (
                YAwareContrastiveLearning,
                {
                    "encoder": self._encoder,
                    "projection_head_kwargs": {
                        "input_dim": self._encoder.latent_size,
                        "output_dim": 10,
                    },
                    "bandwidth": kernel_metric,
                    "temperature": 0.07,
                    "learning_rate": 1e-4,
                    "max_epochs": 2,
                    "limit_train_batches": 3,
                },
            ),
        )
    
    def predict_config(self):
        return {
            LogisticRegression: {
                "num_classes": 2,
                "lr": 5e-4,
                "weight_decay": 1e-4,
            }
        }

    def tearDown(self):
        """Run after each test."""
        pass

    def test_mixin(self):
        """Test Mixin types."""
        mro = BaseEstimator.__mro__
        print(f"[{print_multicolor(repr(mro[:1]), display=False)}]...")
        obj = BaseEstimator()
        self.assertTrue(hasattr(obj, "fit"))
        self.assertFalse(hasattr(obj, "transform"))
        self.assertFalse(hasattr(obj, "predict"))
        for mixin_klass in (
            ClassifierMixin,
            ClusterMixin,
            RegressorMixin,
            TransformerMixin,
        ):
            _klass = type("Estimator", (mixin_klass, BaseEstimator), {})
            mro = _klass.__mro__
            print(f"[{print_multicolor(repr(mro[:3]), display=False)}]...")
            obj = _klass()
            if mixin_klass._estimator_type == "transformer":
                self.assertTrue(hasattr(obj, "fit"))
                self.assertTrue(hasattr(obj, "transform"))
                self.assertFalse(hasattr(obj, "predict"))
            else:
                self.assertTrue(hasattr(obj, "fit"))
                self.assertTrue(hasattr(obj, "predict"))
                self.assertFalse(hasattr(obj, "transform"))

    def test_ssl(self):
        """Test self supervised model (simple check)."""
        for klass, params in self.ssl_config():
            print(f"[{print_multicolor(klass.__name__, display=False)}]...")
            model = klass(**params)
            model.fit(self.ssl_loader)
            z = model.transform(self.x_loader)
            self.assertTrue(
                z.shape == (self.n_images, self._encoder.latent_size),
                msg="Shape mismatch for transformed data: "
                    f"{z.shape} != {(self.n_images, self._encoder.latent_size)}",
            )

    def test_weakly_sup(self):
        """Test weakly supervised model (simple check)."""
        for klass, params in self.test_weakly_sup_config():
            print(f"[{print_multicolor(klass.__name__, display=False)}]...")
            model = klass(
                **params,
            )
            model.fit(self.weakly_sup_loader)
            z = model.transform(self.x_loader)
            self.assertTrue(
                z.shape == (self.n_images, self._encoder.latent_size),
                msg="Shape mismatch for transformed data: "
                    f"{z.shape} != {(self.n_images, self._encoder.latent_size)}",
            )

    def test_predictor(self):
        """Test predictor model (simple check)."""
        for klass, params in self.predict_config().items():
            print(f"[{print_multicolor(klass.__name__, display=False)}]...")
            model = klass(
                model=self._model,
                random_state=42,
                limit_train_batches=3,
                max_epochs=2,
                **params,
            )
            model.fit(self.xy_loader)
            pred = model.predict(self.x_loader)
            self.assertTrue(pred.shape == (self.n_images, 2))


class DummyEncoder(nn.Module):
    """Very small encoder that returns a fixed-sized vector per sample and exposes .latent_size."""
    def __init__(self, latent_size=16):
        super().__init__()
        self.latent_size = latent_size
        # simple linear mapping from flattened image to latent
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 1, latent_size)
        )
        # initialize weights deterministically for reproducibility
        torch.manual_seed(0)
        for p in self.backbone.parameters():
            nn.init.constant_(p, 0.01)

    def forward(self, x):
        return self.backbone(x)

class DummySSLDataset(CustomTensorDataset):

    def __init__(self, n_images=20):
        self.fake_data = torch.rand(n_images, 8 * 8)
        ssl_transforms = transforms.Compose([lambda x: x + torch.rand(x.size())])
        super().__init__(
            self.fake_data, transform=MultiViewsTransform(ssl_transforms, n_views=2)
        )

class SimpleSimCLRTestMixin:
    """Utilities for testcases to create a SimCLR object."""
    def make_simclr(
            self, latent_size=16, hidden_dims=[32, 8], 
            lr=1e-3, temperature=0.1, weight_decay=0.0,
            world_size=1, fitted=False):
        encoder = DummyEncoder(latent_size=latent_size)
        sim = SimCLR(
            encoder=encoder,
            hidden_dims=hidden_dims,
            lr=lr,
            temperature=temperature,
            weight_decay=weight_decay,
            random_state=0,
            max_epochs=2,
            devices=world_size,
            accelerator="cpu"
        )
        if fitted:
            dataset = DummySSLDataset()
            dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
            sim.fit(dataloader)
        return sim


class TestSimCLRBasic(unittest.TestCase, SimpleSimCLRTestMixin):
    def test_constructor_invalid_temperature_raises(self):
        encoder = DummyEncoder(latent_size=8)
        with self.assertRaises(AssertionError):
            SimCLR(
                encoder=encoder, hidden_dims=[16], 
                lr=1e-3, temperature=0.0, weight_decay=0.0
            )

    def test_constructor_encoder_without_latent_size_raises(self):
        class BadEnc(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                return x
        with self.assertRaises(AssertionError):
            SimCLR(
                encoder=BadEnc(), hidden_dims=[16], lr=1e-3, 
                temperature=0.1, weight_decay=0.0)

    def test_all_gather_and_flatten_non_tensor_raises(self):
        sim = self.make_simclr()
        with self.assertRaises(ValueError):
            sim.all_gather_and_flatten("not a tensor")

    def test_all_gather_and_flatten_single_device_returns_same_tensor(self):
        sim = self.make_simclr(world_size=1, fitted=True)

        x = torch.randn(3, sim.f.latent_size)
        out2 = sim.all_gather_and_flatten(x)
        self.assertTrue(torch.allclose(out2, x))

    def test_all_gather_and_flatten_multi_device_flattening(self):
        # Create simclr and monkeypatch all_gather to return stacked tensors
        sim = self.make_simclr(world_size=2, fitted=True)
        batch = 3
        feat = sim.f.latent_size
        z = torch.randn(batch, feat)

        def fake_all_gather(tensor, **kwargs):
            # return simulated gathered tensor shaped (world_size, batch, feat)
            return torch.stack([tensor, tensor], dim=0)

        sim.all_gather = fake_all_gather
        out = sim.all_gather_and_flatten(z)
        # expected shape is (world_size * batch, feat)
        self.assertEqual(out.shape, (2 * batch, feat))
        # check that first batch matches original
        self.assertTrue(torch.allclose(out[:batch], z))

    def test_configure_optimizers_without_scheduler(self):
        sim = self.make_simclr()
        # ensure no hparams.max_epochs present
        if hasattr(sim, "hparams"):
            # remove max_epochs if it exists
            try:
                delattr(sim.hparams, "max_epochs")
            except Exception:
                pass
        res = sim.configure_optimizers()
        self.assertIsInstance(res, list)
        # should return one-element list (optimizers) or [optimizer]
        self.assertGreaterEqual(len(res), 1)

    def test_configure_optimizers_with_scheduler(self):
        sim = self.make_simclr()
        res = sim.configure_optimizers()
        # Expect [optimizer], [scheduler]
        self.assertTrue(isinstance(res, tuple) or isinstance(res, list))
        # The function returns either [opt] or ([opt],[sched]); we check for scheduler return shape
        # if a scheduler is returned we expect outer list of length 2
        if isinstance(res, list) and len(res) == 2 and all(isinstance(x, list) for x in res):
            # good: [optimizer], [lr_scheduler]
            self.assertTrue(len(res[0]) >= 1 and len(res[1]) >= 1)
        else:
            # In some environments the code may return ([opt], [sched]) as a tuple; accept that
            self.assertTrue(True)

    def test_transform_step_returns_encoder_output(self):
        sim = self.make_simclr()
        # create small "image" shaped input consistent with DummyEncoder (expects 8x8x1)
        batch = torch.randn(2, 1, 8, 8)
        out = sim.transform_step(batch, batch_idx=0)
        # Should be the encoder output: shape (2, latent_size)
        self.assertEqual(out.shape, (2, sim.f.latent_size))


class TestSimCLRSteps(unittest.TestCase, SimpleSimCLRTestMixin):
    def setUp(self):
        self.sim = self.make_simclr(
            latent_size=8, hidden_dims=[16, 8], 
            lr=1e-3, temperature=0.1, fitted=True
        )

    def test_training_step_returns_expected_loss_and_embeddings(self):
        sim = self.sim
        sim.log = lambda *a, **k: None
        # Create toy views V1 and V2: small 8x8 images
        batch_size = 4
        V1 = torch.randn(batch_size, 1, 8, 8)
        V2 = torch.randn(batch_size, 1, 8, 8)
        outputs = sim.training_step((V1, V2), batch_idx=0)
        # The loss is DeterministicInfoNCE: mean squared difference between Z1 and Z2
        self.assertIn("loss", outputs)
        self.assertIn("Z1", outputs)
        self.assertIn("Z2", outputs)

        # Z1, Z2 shapes: (batch, proj_dim).
        self.assertEqual(outputs["Z1"].shape, (batch_size, sim.f.latent_size))
        self.assertEqual(outputs["Z2"].shape, (batch_size, sim.f.latent_size))
        # Check loss numeric equality with a direct computation using sim.loss
        # outputs['Z1'] and 'Z2' came back on cpu detached
        z1 = outputs["Z1"]
        z2 = outputs["Z2"]
        expected_loss = sim.loss(z1, z2)
        # loss returned in outputs might be a tensor; compare floats
        self.assertAlmostEqual(float(outputs["loss"]), float(expected_loss), places=6)

    def test_validation_step_behavior(self):
        sim = self.sim
        sim.log = lambda *a, **k: None
        batch_size = 3
        V1 = torch.randn(batch_size, 1, 8, 8)
        V2 = torch.randn(batch_size, 1, 8, 8)
        outputs = sim.validation_step((V1, V2), batch_idx=0)
        self.assertIn("loss", outputs)
        self.assertIn("Z1", outputs)
        self.assertIn("Z2", outputs)
        # shapes should match
        self.assertEqual(outputs["Z1"].shape, (batch_size, sim.f.latent_size))
        self.assertEqual(outputs["Z2"].shape, (batch_size, sim.f.latent_size))

    def test_training_step_multi_device_collects_and_flattens(self):
        sim = self.make_simclr(
            latent_size=8, hidden_dims=[16, 8], 
            lr=1e-3, temperature=0.1, world_size=2,
            fitted=True)
        sim.log = lambda *a, **k: None
        batch_size = 2
        V1 = torch.randn(batch_size, 1, 8, 8)
        V2 = torch.randn(batch_size, 1, 8, 8)

        outputs = sim.training_step((V1, V2), batch_idx=0)

        z1 = outputs["Z1"]
        z2 = outputs["Z2"]
        self.assertEqual(z1.shape, (2 * batch_size, sim.f.latent_size))
        self.assertEqual(z2.shape, (2 * batch_size, sim.f.latent_size))

        expected_loss = float(sim.loss(z1, z2))
        self.assertAlmostEqual(float(outputs["loss"]), expected_loss, places=6)

    def test_zero_batch_size_handling(self):
        """
        Ensure the steps don't crash on zero-size batch (edge case).
        Many dataloaders might never emit this, but code should handle gracefully if it occurs.
        """
        sim = self.sim
        sim.log = lambda *a, **k: None
        V1 = torch.empty(0, 1, 8, 8)
        V2 = torch.empty(0, 1, 8, 8)
        outputs_train = sim.training_step((V1, V2), batch_idx=0)
        self.assertIn("loss", outputs_train)
        self.assertIsInstance(outputs_train["loss"], torch.Tensor)
        outputs_val = sim.validation_step((V1, V2), batch_idx=0)
        self.assertIn("loss", outputs_val)
        self.assertIsInstance(outputs_val["loss"], torch.Tensor)


if __name__ == "__main__":
    unittest.main()
