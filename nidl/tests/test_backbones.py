##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import unittest

import torch

from nidl.backbones.volume import (
    AlexNet,
    VisionTransformer3D,
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
            self.n_images, self.n_channels, 128, 128, 128
        )

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


class TestVisionTransformer3D(unittest.TestCase):
    """ Test the 3D ViT: dynamic input sizes and patch masking.
    """
    def setUp(self):
        torch.manual_seed(0)
        self.dim = 16

    def make_vit(self, **kwargs):
        """ Tiny ViT built for 32^3 inputs (2x2x2 patches of 16^3).
        """
        params = {
            "img_size": 32, 
            "patch_size": 16, 
            "embed_dim": self.dim,
            "depth": 1, 
            "num_heads": 2
        }
        params.update(kwargs)
        return VisionTransformer3D(**params).eval()

    def test_same_size_unchanged(self):
        """ At the built size, dynamic and strict models give the same output.
        """
        strict = self.make_vit()
        dynamic = self.make_vit(dynamic_img_size=True)
        dynamic.load_state_dict(strict.state_dict())
        x = torch.rand(2, 1, 32, 32, 32)
        self.assertTrue(torch.equal(strict(x), dynamic(x)))

    def test_dynamic_sizes(self):
        """ Mixed (also non-cubic) input sizes work for all pos_embed modes.
        """
        for pos_embed, no_embed_class in (("learned", False), ("sincos", False),
                                          ("none", False), ("learned", True)):
            vit = self.make_vit(
                dynamic_img_size=True, 
                pos_embed=pos_embed,
                no_embed_class=no_embed_class
            )
            for size in ((16, 16, 16), (48, 48, 48), (32, 48, 16)):
                x = torch.rand(2, 1, *size)
                n_patches = (size[0] // 16) * (size[1] // 16) * (size[2] // 16)
                tokens = vit.forward_features(x)
                self.assertEqual(tokens.shape, (2, 1 + n_patches, self.dim))
                self.assertEqual(vit(x).shape, (2, self.dim))
                _, inter = vit.forward_intermediates(x)
                self.assertEqual(inter[-1].shape, tokens.shape)

    def test_strict_size_raises(self):
        """ Without dynamic_img_size, a different input size raises an error.
        """
        vit = self.make_vit()
        with self.assertRaises(ValueError):
            vit(torch.rand(2, 1, 16, 16, 16))

    def test_resize_pos_embed(self):
        """ Class token is not resized
        """
        vit = self.make_vit(dynamic_img_size=True)
        pos = vit._resize_pos_embed((3, 1, 2))
        self.assertEqual(pos.shape, (1, 1 + 6, self.dim))
        self.assertTrue(torch.equal(pos[:, 0], vit.pos_embed[:, 0]))

if __name__ == "__main__":
    unittest.main()