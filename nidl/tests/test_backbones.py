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
    densenet121,
    resnet18,
    resnet18_trunc,
    resnet50,
    resnet50_trunc,
    DynamicSizeViT,
    VisionTransformer3DMoE,
    MoEParams
)

from nidl.utils import print_multicolor

VOLUME_SHAPE = 128
PATCH_SIZE = 16
TOKENS_COUNT = (128 // 16) * (128 // 16) * (128 // 16)
SMALLER_SHAPE = 96
TOKENS_COUNT_SMALLER = (96 // 16) * (96 // 16) * (96 // 16)
BLOCKS_NUMBER = 1

class TestBackbones(unittest.TestCase):
    """ Test backbones.
    """
    def setUp(self):
        """ Setup test.
        """
        self.n_images = 3
        self.n_channels = 1
        self.fake_data = torch.rand(
            self.n_images, 
            self.n_channels, 
            VOLUME_SHAPE, 
            VOLUME_SHAPE, 
            VOLUME_SHAPE
        )
        self.fake_data_diff_shape = torch.rand(
            self.n_images,
            self.n_channels,
            SMALLER_SHAPE,
            SMALLER_SHAPE,
            SMALLER_SHAPE
        )

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def cnn_config(self):
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

    def _tiny_vit_moe(self):
        """
            A minimal VisionTransformer3DMoE small enough to run instantly on CPU.
        """
        moe_params = MoEParams(
            dim=12,
            n_shared_experts=1,
            n_routed_experts=2,
            n_activated_experts=1,
            moe_inter_dim=4,
            moe_layer_indices=(0,),
        )

        return VisionTransformer3DMoE(
            img_size=(VOLUME_SHAPE, VOLUME_SHAPE, VOLUME_SHAPE),
            patch_size=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
            in_chans=1,
            embed_dim=12,
            depth=BLOCKS_NUMBER,
            num_heads=2,
            use_moe=True,
            moe_params=moe_params,
            class_token=False,
            reg_tokens=0
        )

    def _tiny_dynamic_vit(self):
        """
            A minimal DynamicSizeViT small enough to run instantly on CPU.
        """

        return DynamicSizeViT(
            img_size=(VOLUME_SHAPE, VOLUME_SHAPE, VOLUME_SHAPE),
            patch_size=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
            in_chans=1,
            embed_dim=12,
            depth=BLOCKS_NUMBER,
            num_heads=2,
            class_token=False,
            reg_tokens=0,
            dynamic_img_size=True
        )

    def test_cnn_backbones(self):
        """ 
            Test CNN volume backbones (simple check).
        """
        for klass, params in self.cnn_config().items():
            print(f"[{print_multicolor(klass.__name__, display=False)}]...")
            backbone = klass(**params)
            out = backbone(self.fake_data)
            if "_trunc" not in klass.__name__:
                self.assertTrue(out.shape == (self.n_images, 10))
            else:
                self.assertTrue(out.shape == (self.n_images, 64, 32, 32, 32))
    
    def test_vitmoe_forward_features(self):
        """
            Test shape produced by forward_features method of MOE Vit
        """
        print(f"[{print_multicolor('ViT MoE forward_features', display=False)}]...")
        backbone = self._tiny_vit_moe()
        out = backbone.forward_features(self.fake_data)

        self.assertTrue(len(out) == 2)
        self.assertTrue(out[0].shape == (self.n_images, TOKENS_COUNT, backbone.embed_dim))
        self.assertTrue(len(out[1]) == BLOCKS_NUMBER)
    
    def test_dynamicvit_forward_features(self):
        """
            Test shape produced by forward_features method of DynamicSizeVit
        """
        print(f"[{print_multicolor('DynamicSizeVit forward_features', display=False)}]...")
        backbone = self._tiny_dynamic_vit()
        out_larger = backbone.forward_features(self.fake_data)
        out_smaller = backbone.forward_features(self.fake_data_diff_shape)

        self.assertTrue(out_larger.shape == (self.n_images, TOKENS_COUNT, backbone.embed_dim))
        self.assertTrue(out_smaller.shape == (self.n_images, TOKENS_COUNT_SMALLER, backbone.embed_dim))


if __name__ == "__main__":
    unittest.main()
