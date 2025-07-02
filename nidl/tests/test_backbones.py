##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import unittest

import torch

from nidl.volume.backbones import (
    AlexNet,
    densenet121,
    resnet18,
    resnet18_trunc,
    resnet50,
    resnet50_trunc,
)
from nidl.utils import print_multicolor


class TestBackbones(unittest.TestCase):
    """ Test backbones.
    """
    def setUp(self):
        """ Setup test.
        """
        self.n_images = 3
        self.n_channels = 2
        self.fake_data = torch.rand(
            self.n_images, self.n_channels, 128, 128, 128)

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def volume_config(self):
        return {
            AlexNet: {
                "n_embedding": 10,
                "in_channels": self.n_channels
            },
            resnet18: {
                "n_embedding": 10,
                "in_channels": self.n_channels
            },
            resnet18_trunc: {
                "n_embedding": 10,
                "in_channels": self.n_channels,
                "depth": 0
            },
            resnet50: {
                "n_embedding": 10,
                "in_channels": self.n_channels
            },
            resnet50_trunc: {
                "n_embedding": 10,
                "in_channels": self.n_channels,
                "depth": 0
            },
            densenet121: {
                "n_embedding": 10,
                "in_channels": self.n_channels
            }
        }

    def test_volume_backbones(self):
        """ Test volume backbones (simple check).
        """
        for klass, params in self.volume_config().items():
            print(f"[{print_multicolor(klass.__name__, display=False)}]...")
            backbone = klass(**params)
            out = backbone(self.fake_data)
            if "_trunc" not in klass.__name__:
                self.assertTrue(out.shape == (self.n_images, 10))
            else:
                self.assertTrue(out.shape == (self.n_images, 64, 32, 32, 32))


if __name__ == "__main__":
    unittest.main()
