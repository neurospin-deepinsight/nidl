##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from collections import OrderedDict

import unittest

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from nidl.estimators import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    RegressorMixin,
    TransformerMixin,
)
from nidl.estimators.ssl import (
    BarlowTwins,
    SimCLR,
    YAwareContrastiveLearning,
)
from nidl.estimators.ssl.dino import DINO
from nidl.estimators.ssl.utils.projection_heads import (
    BarlowTwinsProjectionHead,
    SimCLRProjectionHead,
)
from nidl.estimators.autoencoders import VAE
from nidl.estimators.linear import LogisticRegression
from nidl.losses.yaware_infonce import KernelMetric
from nidl.losses.beta_vae import BetaVAELoss
from nidl.transforms import MultiViewsTransform
from nidl.utils import print_multicolor


class CustomTensorDataset(Dataset):
    """ TensorDataset with support of transforms.
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
    

class TestEstimators(unittest.TestCase):
    """ Test estimators: simple checks.
    """
    def setUp(self):
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
        multicrop_ssl_dataset = CustomTensorDataset(
            self.fake_data, transform=MultiViewsTransform(ssl_transforms, n_views=6)
        )
        x_dataset = CustomTensorDataset(self.fake_data)
        xy_dataset = CustomTensorDataset(self.fake_data, labels=self.fake_labels)
        xxy_dataset = CustomTensorDataset(
            self.fake_data, labels=self.fake_labels, 
            transform=MultiViewsTransform(ssl_transforms, n_views=2)
        )
        self.ssl_loader = DataLoader(ssl_dataset, batch_size=2, shuffle=False)
        self.multicrop_ssl_loader = DataLoader(multicrop_ssl_dataset, batch_size=2, shuffle=False)
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
                BarlowTwins,
                {
                    "encoder": self._encoder,
                    "projection_head": BarlowTwinsProjectionHead(
                        input_dim=self._encoder.latent_size, 
                        hidden_dim=3, 
                        output_dim=3
                    ),
                    "projection_head_kwargs": { # ignored
                        "input_dim": self._encoder.latent_size,
                        "output_dim": 3,
                    },
                    "learning_rate": 5e-4,
                    "lambd": 0.005,
                    "optimizer_kwargs": {"weight_decay": 1e-4},
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

    def multicrop_ssl_config(self):
        return ((
            DINO,
                {
                    "encoder": self._encoder,
                    "proj_input_dim": self._encoder.latent_size,
                    "proj_hidden_dim": 3,
                    "proj_output_dim": 3,
                    "warmup_teacher_temp": 0.04,
                    "teacher_temperature": 0.07,
                    "warmup_teacher_temp_epochs": 30,
                    "student_temperature": 0.1,
                    "num_local_crops": 4,
                    "max_epochs": 2,
                }
            ),
            (
            DINO,
                {
                    "encoder": self._encoder,
                    "proj_input_dim": self._encoder.latent_size,
                    "proj_hidden_dim": 3,
                    "proj_output_dim": 3,
                    "warmup_teacher_temp": 0.07,
                    "teacher_temperature": 0.07,
                    "warmup_teacher_temp_epochs": 0,
                    "student_temperature": 0.1,
                    "lr_scheduler": "none",
                    "num_local_crops": 4,
                    "max_epochs": 2,
                }
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
    
    def vae_config(self):
        return {
            VAE:
                {
                    "encoder": DummyEncoder(),
                    "decoder": DummyDecoder(),
                    "encoder_out_dim": 4,
                    "latent_dim": 2,
                    "beta": 1.0,
                    "stochastic_transform": True,
                    "lr": 1e-3,
                    "weight_decay": 0.0,
                    "max_epochs": 2,
                }
        }

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
        """ Test self supervised model (simple check)."""
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

    def test_multicrop_ssl(self):
        """ Test self supervised model with multi-crop (simple check)."""
        for klass, params in self.multicrop_ssl_config():
            print(f"[{print_multicolor(klass.__name__, display=False)}]...")
            model = klass(**params)
            model.fit(self.multicrop_ssl_loader)
            z = model.transform(self.x_loader)
            self.assertTrue(
                z.shape == (self.n_images, self._encoder.latent_size),
                msg="Shape mismatch for transformed data: "
                    f"{z.shape} != {(self.n_images, self._encoder.latent_size)}",
            )
    
    def test_vae(self):
        """ Test VAE model (simple check).
        """
        for klass, params in self.vae_config().items():
            print(f"[{print_multicolor(klass.__name__, display=False)}]...")
            model = klass(**params)
            model.fit(self.x_loader)
            z = model.transform(self.x_loader)
            self.assertTrue(
                z.shape == (self.n_images, model.latent_dim),
                msg="Shape mismatch for transformed data: "
                    f"{z.shape} != {(self.n_images, model.latent_dim)}",
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
    def __init__(self, in_dim=25, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class DummyDecoder(nn.Module):
    def __init__(self, latent_dim=2, out_dim=25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class TestVAE(unittest.TestCase):
    """ Test VAE estimator.
    """
    def setUp(self):
        encoder = DummyEncoder()
        decoder = DummyDecoder()
        self.vae = VAE(
            encoder=encoder,
            decoder=decoder,
            encoder_out_dim=4,
            latent_dim=2,
            beta=1.0,
            stochastic_transform=True,
            lr=1e-3,
            weight_decay=0.0,
        )
        self.batch = torch.rand(2, 25)

    def test_init_attributes(self):
        self.assertIsInstance(self.vae.encoder, nn.Module)
        self.assertIsInstance(self.vae.decoder, nn.Module)
        self.assertIsInstance(self.vae.fc_mu, nn.Linear)
        self.assertIsInstance(self.vae.fc_logvar, nn.Linear)
        self.assertIsInstance(self.vae.criterion, BetaVAELoss)
        self.assertEqual(self.vae.latent_dim, 2)
        self.assertEqual(self.vae.beta, 1.0)

    def test_forward_shape(self):
        z = self.vae.forward(self.batch)
        self.assertEqual(z.shape, (self.batch.size(0), self.vae.latent_dim))
        self.assertTrue(z.requires_grad)

    def test_sampling(self):
        x = self.vae.sample(n_samples=4)
        self.assertEqual(x.shape, (4, 25))
        self.assertTrue(torch.isfinite(x).all())

    def test_run_step_and_losses(self):
        x_hat, q = self.vae._run_step(self.batch)
        self.assertEqual(x_hat.shape[0], self.batch.shape[0])
        self.assertTrue(hasattr(q, "loc"))

        losses = self.vae.training_step(self.batch, batch_idx=0)
        for key in ("loss", "rec_loss", "kl_loss"):
            self.assertIn(key, losses)
            self.assertIsInstance(losses[key], torch.Tensor)
            self.assertEqual(losses[key].ndim, 0)

    def test_validation_step_returns_losses(self):
        losses = self.vae.validation_step(self.batch, batch_idx=0)
        self.assertEqual(set(losses), {"loss", "rec_loss", "kl_loss"})

    def test_transform_step_stochastic_and_deterministic(self):
        for stochastic in (True, False):
            vae = VAE(
                encoder=DummyEncoder(),
                decoder=DummyDecoder(),
                encoder_out_dim=4,
                latent_dim=2,
                stochastic_transform=stochastic,
            )
            x = torch.rand(3, 25)
            z = vae.transform_step(x, batch_idx=0)
            self.assertEqual(z.shape, (3, 2))
            if not stochastic:
                with torch.no_grad():
                    z_mu = vae.fc_mu(vae.encoder(x))
                    self.assertTrue(torch.allclose(z, z_mu, atol=1e-6))

    def test_configure_optimizers_returns_adamw(self):
        opt_list = self.vae.configure_optimizers()
        self.assertIsInstance(opt_list, list)
        self.assertIsInstance(opt_list[0], torch.optim.AdamW)

    def test_backward_and_optimizer_step(self):
        opt = self.vae.configure_optimizers()[0]
        opt.zero_grad()
        losses = self.vae.training_step(self.batch, batch_idx=0)
        losses["loss"].backward()

        before = [p.detach().clone() for p in self.vae.parameters()]
        opt.step()
        after = list(self.vae.parameters())
        self.assertTrue(any((a - b).abs().sum() > 0 for a, b in zip(after, before)))


if __name__ == "__main__":
    unittest.main()
