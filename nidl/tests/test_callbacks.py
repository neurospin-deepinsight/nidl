##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from collections import OrderedDict

import unittest

import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from nidl.callbacks.caching import CachingCallback
from nidl.callbacks.check_typing import BatchTypingCallback
from nidl.estimators.linear import LogisticRegression
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
        self.n_images = 20
        self.fake_data = torch.rand(self.n_images, 5 * 5)
        self.fake_labels = torch.randint(1, (self.n_images, ))
        x_dataset = CustomTensorDataset(
            self.fake_data
        )
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

    def test_callbacks(self):
        """ Test callbacks (simple check).
        """
        model = LogisticRegression(
            model=self._model,
            random_state=42,
            limit_train_batches=3,
            max_epochs=2,
            num_classes=2,
            lr=5e-4,
            weight_decay=1e-4,
            callbacks=[
                BatchTypingCallback(),
                CachingCallback(train_dataloader=self.x_loader,
                                validation_dataloader=None,
                                test_dataloader=None,
                                frequency=1,
                                model_attr="model.encoder",
                                auxiliaries=self.fake_labels),
            ]
        )
        model.fit(self.xy_loader, val_dataloader=self.xy_loader)
        self.assertTrue(model.embeddings.last_epoch == 1)
        self.assertTrue(
            np.allclose(model.embeddings.auxiliaries, self.fake_labels))
        self.assertTrue(model.embeddings.train.shape == (
            self.n_images, self._encoder.latent_size))
        self.assertTrue(model.embeddings.validation is None)
        self.assertTrue(model.embeddings.test is None)
        pred = model.predict(self.x_loader)
        self.assertTrue(pred.shape == (self.n_images, 2))


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
