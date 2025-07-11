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

from nidl.transforms import ContrastiveTransforms
from nidl.callbacks.check_typing import BatchTypingCallback
from nidl.callbacks.model_probing import LogisticRegressionCVCallback, \
    KNeighborsClassifierCVCallback, KNeighborsRegressorCVCallback, \
    RidgeCVCallback, ModelProbing
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
        self._fc = nn.Linear(self._encoder.latent_size, 2)
        self._model =  nn.Sequential(OrderedDict([
            ("encoder", self._encoder),
            ("fc", self._fc)
        ]))
        self.n_images = 30
        self.fake_data = torch.rand(self.n_images, 5 * 5)
        self.fake_labels = torch.randint(0, 3, (self.n_images, ))
        self.fake_continuous_labels = torch.rand(self.n_images, 1)
        self.fake_multivariate_continuous_labels = torch.rand(self.n_images, 3)
        ssl_transforms = transforms.Compose([
            lambda x: x + torch.rand(x.size())
        ])
        x_dataset = CustomTensorDataset(
            self.fake_data
        )
        # multi-class classification dataset
        xy_dataset = CustomTensorDataset( 
            self.fake_data,
            labels=self.fake_labels
        )
        # regression dataset
        xy_reg_dataset = CustomTensorDataset( 
            self.fake_data,
            labels=self.fake_continuous_labels
        )
        # multivariate regression dataset
        xy_multivariate_reg_dataset = CustomTensorDataset( 
            self.fake_data,
            labels=self.fake_multivariate_continuous_labels
        )
        ssl_dataset = CustomTensorDataset(
            self.fake_data,
            transform=ContrastiveTransforms(ssl_transforms, n_views=2)
        )
        self.x_loader = DataLoader(x_dataset, batch_size=2, shuffle=False)
        self.xy_loader = DataLoader(xy_dataset, batch_size=2, shuffle=False)
        self.xy_reg_loader = DataLoader(
            xy_reg_dataset, batch_size=2, shuffle=False)
        self.xy_multivariate_reg_loader = DataLoader(
            xy_multivariate_reg_dataset, batch_size=2, shuffle=False)
        self.ssl_loader = DataLoader(ssl_dataset, batch_size=2, shuffle=False)

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
            ]
        )
        model.fit(self.xy_loader)
        pred = model.predict(self.x_loader)
        self.assertTrue(pred.shape == (self.n_images, ))
    
    def test_ridgecv_reg_probing_callback(self):
        """ Test RidgeCV regression probing callback. """
        model = SimCLR(
            encoder=self._encoder,
            random_state=42,
            limit_train_batches=3,
            max_epochs=2,
            temperature=0.1,
            hidden_dims=[64, 32],
            lr=5e-4,
            weight_decay=1e-4,
            callbacks=[
                RidgeCVCallback(
                    train_dataloader=self.xy_reg_loader,
                    test_dataloader=self.xy_reg_loader,
                    probe_name="ridge",
                    cv=3,
                    every_n_train_epochs=1,
                    every_n_val_epochs=1,
                    prog_bar=True,
                    on_test_epoch_start=True,
                    on_test_epoch_end=True
                )
            ],
            enable_checkpointing=False
        )
        model.fit(self.ssl_loader, val_dataloader=self.ssl_loader)
        z = model.transform(self.x_loader)
        self.assertTrue(z.shape == (self.n_images, self._encoder.latent_size),
                        msg="Predicted shape mismatch: "
                            f"expected {(self.n_images, self._encoder.latent_size)}, "
                            f"got {z.shape}")

    def test_kneighbors_reg_probing_callback(self):
        """ Test KNeighborsRegressorCV regression probing callback. """
        model = SimCLR(
            encoder=self._encoder,
            random_state=42,
            limit_train_batches=3,
            max_epochs=2,
            temperature=0.1,
            hidden_dims=[64, 32],
            lr=5e-4,
            weight_decay=1e-4,
            callbacks=[
                KNeighborsRegressorCVCallback(
                    train_dataloader=self.xy_reg_loader,
                    test_dataloader=self.xy_reg_loader,
                    cv=5
                )
            ],
            enable_checkpointing=False
        )
        model.fit(self.ssl_loader, val_dataloader=self.ssl_loader)
        z = model.transform(self.x_loader)
        self.assertTrue(z.shape == (self.n_images, self._encoder.latent_size),
                        msg="Predicted shape mismatch: "
                            f"expected {(self.n_images, self._encoder.latent_size)}, "
                            f"got {z.shape}")

    def test_ridgecv_multivariate_reg_probing_callback(self):
        """ Test RidgeCV regression probing callback on multivariate targets. """
        model = SimCLR(
            encoder=self._encoder,
            random_state=42,
            limit_train_batches=3,
            max_epochs=2,
            temperature=0.1,
            hidden_dims=[64, 32],
            lr=5e-4,
            weight_decay=1e-4,
            callbacks=[
                RidgeCVCallback(
                    train_dataloader=self.xy_reg_loader,
                    test_dataloader=self.xy_reg_loader,
                    cv=3
                )
            ],
            enable_checkpointing=False
        )
        model.fit(self.ssl_loader, val_dataloader=self.ssl_loader)
        z = model.transform(self.x_loader)
        self.assertTrue(z.shape == (self.n_images, self._encoder.latent_size),
                        msg="Predicted shape mismatch: "
                            f"expected {(self.n_images, self._encoder.latent_size)}, "
                            f"got {z.shape}")

    def test_kneighbors_multivariate_reg_probing_callback(self):
        """ Test KNeighborsRegressorCV regression probing callback on multivariate targets. """
        model = SimCLR(
            encoder=self._encoder,
            random_state=42,
            limit_train_batches=3,
            max_epochs=2,
            temperature=0.1,
            hidden_dims=[64, 32],
            lr=5e-4,
            weight_decay=1e-4,
            callbacks=[
                KNeighborsRegressorCVCallback(
                    train_dataloader=self.xy_multivariate_reg_loader,
                    test_dataloader=self.xy_multivariate_reg_loader,
                    cv=5
                )
            ],
            enable_checkpointing=False
        )
        model.fit(self.ssl_loader, val_dataloader=self.ssl_loader)
        z = model.transform(self.x_loader)
        self.assertTrue(z.shape == (self.n_images, self._encoder.latent_size),
                        msg="Predicted shape mismatch: "
                            f"expected {(self.n_images, self._encoder.latent_size)}, "
                            f"got {z.shape}")

    def test_logistic_regressioncv_classif_probing_callback(self):
        """ Test LogisticRegressionCV classification probing callback.
        """
        model = SimCLR(
            encoder=self._encoder,
            random_state=42,
            limit_train_batches=3,
            max_epochs=2,
            temperature=0.1,
            hidden_dims=[64, 32],
            lr=5e-4,
            weight_decay=1e-4,
            callbacks=[
                LogisticRegressionCVCallback(
                    train_dataloader=self.xy_loader,
                    test_dataloader=self.xy_loader,
                    probe_name="logistic",
                    cv=3
                )
            ],
            enable_checkpointing=False

        )
        model.fit(self.ssl_loader, val_dataloader=self.ssl_loader)
        z = model.transform(self.x_loader)
        self.assertTrue(z.shape == (self.n_images, self._encoder.latent_size),
                        msg="Predicted shape mismatch: "
                            f"expected {(self.n_images, self._encoder.latent_size)}, "
                            f"got {z.shape}")

    def test_kneighbors_classif_probing_callback(self):
        """ Test KNeighborsClassifierCV classification probing callback.
        """
        model = SimCLR(
            encoder=self._encoder,
            random_state=42,
            limit_train_batches=3,
            max_epochs=2,
            temperature=0.1,
            hidden_dims=[64, 32],
            lr=5e-4,
            weight_decay=1e-4,
            callbacks=[
                KNeighborsClassifierCVCallback(
                    train_dataloader=self.xy_loader,
                    test_dataloader=self.xy_loader,
                    probe_name="knn",
                    cv=5
                )
            ],
            enable_checkpointing=False
        )
        model.fit(self.ssl_loader, val_dataloader=self.ssl_loader)
        z = model.transform(self.x_loader)
        self.assertTrue(z.shape == (self.n_images, self._encoder.latent_size),
                        msg="Predicted shape mismatch: "
                            f"expected {(self.n_images, self._encoder.latent_size)}, "
                            f"got {z.shape}")


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
