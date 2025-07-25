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
from nidl.estimators.ssl import SimCLR, YAwareContrastiveLearning
from nidl.estimators.ssl.utils.projection_heads import SimCLRProjectionHead
from nidl.estimators.linear import LogisticRegression
from nidl.losses.yaware_infonce import KernelMetric
from nidl.transforms import MultiViewsTransform
from nidl.utils import print_multicolor


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


if __name__ == "__main__":
    unittest.main()
