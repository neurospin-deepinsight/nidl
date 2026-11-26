##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import copy
import json
import pickle
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml
from huggingface_hub.utils import GatedRepoError
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from torchvision import transforms as transforms_v1
from torchvision.transforms import v2 as transforms_v2

from nidl.backbones.volume.densenet3d import densenet121
from nidl.backbones.volume.resnet3d import resnet18
from nidl.backbones.volume.vit3d import VisionTransformer3D
from nidl.backbones.volume.vit3d_moe import (
    MoEParams,
    RoPEAttention,
    VisionTransformer3DMoE,
)
from nidl.estimators import BaseEstimator
from nidl.estimators.dummy import DummyEmbeddingEstimator
from nidl.estimators.external import (
    ExternalWeights,
    NeuroJEPAWeights,
    get_converter,
)
from nidl.estimators.ssl import NeuroJEPA, SimCLR, YAwareContrastiveLearning
from nidl.estimators.ssl.neurojepa import MaskScaleConfig
from nidl.estimators.ssl.utils.projection_heads import SimCLRProjectionHead
from nidl.transforms import Transform
from nidl.transforms.volume.preprocessing import (
    BrainIACPreprocessing,
    CropForeground,
    CropOrPad,
    NeuroJEPAPreprocessing,
    NeuroSimCLRPreprocessing,
    Reorient,
    ZNormalization,
)
from nidl.utils.config import (
    ConfigMixin,
    check_config,
    get_config,
    instantiate,
)
from nidl.utils.hub import _write_config, push_to_hub, save_pretrained

TINY_SHAPE = (1, 24, 24, 24)
AUTHORS = "authors/tiny-jepa"


def _neuro_jepa(embed_dim, depth, num_heads, inter_dim):
    def build():
        return NeuroJEPA(
            encoder=VisionTransformer3DMoE(
                img_size=(96, 108, 96),
                patch_size=(12, 12, 12),
                in_chans=1,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                use_moe=True,
                moe_params={
                    "dim": embed_dim,
                    "n_shared_experts": 2,
                    "n_routed_experts": 16,
                    "n_activated_experts": 6,
                    "moe_inter_dim": inter_dim,
                    "score_func": "softmax",
                    "route_scale": 4.0,
                    "bias_clip": 0.3,
                    "bias_update_rate": 0.0001,
                    "moe_layer_indices": list(range(1, depth, 2)),
                },
                attn_gate=True,
                norm_eps=1e-06,
            ),
            use_moe=True,
            predictor_embed_dim=384,
            predictor_depth=12,
            predictor_num_heads=12,
            foreground_aware=True,
            moe_bias_clip=0.3,
            moe_bias_update_rate=0.0001,
        )

    return build


#: The published models: their estimator, their preprocessing and whether
#: their weights are gated.
PUBLISHED = {
    "yaware-densenet121-bhb10k": {
        "estimator": lambda: YAwareContrastiveLearning(
            encoder=densenet121(n_embedding=None),
            proj_input_dim=1024,
            proj_hidden_dim=512,
            proj_output_dim=128,
            bandwidth=25.0,
            temperature=0.1,
            learning_rate=1e-4,
            weight_decay=5e-5,
        ),
        "preprocessing": ZNormalization,
        "gated": False,
    },
    "neuro-simclr-resnet18": {
        "estimator": lambda: SimCLR(
            encoder=resnet18(
                n_embedding=None, initial_stride=1, downsample_bias=True
            ),
            proj_input_dim=512,
            proj_hidden_dim=512,
            proj_output_dim=64,
            proj_bias=False,
            temperature=0.5,
        ),
        "preprocessing": lambda: NeuroSimCLRPreprocessing(
            input_orientation="RAS"
        ),
        "gated": False,
    },
    "brainiac-vit-b": {
        "estimator": lambda: SimCLR(
            encoder=VisionTransformer3D(
                img_size=96,
                patch_size=16,
                class_token=False,
                global_pool="avg",
                fc_norm=False,
                no_embed_class=True,
                qkv_bias=False,
            ),
            proj_input_dim=768,
            proj_hidden_dim=768,
            proj_output_dim=2048,
            proj_batch_norm=True,
            temperature=0.1,
        ),
        "preprocessing": BrainIACPreprocessing,
        "gated": False,
    },
    "neuro-jepa-vit-b": {
        "estimator": _neuro_jepa(768, 12, 12, 384),
        "preprocessing": lambda: NeuroJEPAPreprocessing(
            input_orientation="RAS"
        ),
        "gated": True,
    },
    "neuro-jepa-vit-l": {
        "estimator": _neuro_jepa(1024, 24, 16, 512),
        "preprocessing": lambda: NeuroJEPAPreprocessing(
            input_orientation="RAS"
        ),
        "gated": True,
    },
}


def _official_encoder(name):
    return PUBLISHED[name]["estimator"]().context_encoder.vit


def _sizes(state_dict):
    floats = sum(
        v.numel() for v in state_dict.values() if v.is_floating_point()
    )
    integers = sum(
        v.numel() for v in state_dict.values() if not v.is_floating_point()
    )
    return floats, integers


def _tuples(value):
    if isinstance(value, dict):
        return {k: _tuples(v) for k, v in value.items()}
    if isinstance(value, list):
        return tuple(_tuples(v) for v in value)
    return value


def _tiny_vit(**kwargs):
    return VisionTransformer3D(
        img_size=16, patch_size=8, embed_dim=12, depth=1, num_heads=2,
        **kwargs,
    )


def _tiny_simclr(fitted=True, **kwargs):
    model = SimCLR(
        encoder=_tiny_vit(),
        proj_input_dim=12,
        proj_hidden_dim=8,
        proj_output_dim=4,
        accelerator="cpu",
        **kwargs,
    )
    model.fitted_ = fitted
    return model


def _tiny_neurojepa(**kwargs):
    """A tiny NeuroJEPA, as the published ones."""
    return NeuroJEPA(
        encoder=VisionTransformer3DMoE(
            img_size=TINY_SHAPE[1:],
            patch_size=(12, 12, 12),
            embed_dim=12,
            depth=2,
            num_heads=2,
            use_moe=True,
            moe_params={
                "dim": 12,
                "n_routed_experts": 4,
                "n_activated_experts": 2,
                "moe_inter_dim": 8,
                "moe_layer_indices": [1],
            },
            attn_gate=True,
            norm_eps=1e-6,
        ),
        use_moe=True,
        predictor_embed_dim=8,
        predictor_depth=1,
        predictor_num_heads=2,
        accelerator="cpu",
        **kwargs,
    )


def _read(path):
    return yaml.safe_load(Path(path).read_text())


class TestOfficialArchitecture(unittest.TestCase):
    """The checkpoints themselves are gated: the architecture of the nidl
    configurations is pinned against what the Hub and the official source
    code publicly state."""

    def test_size_of_published_tensors(self):
        """Exact counts given by the Hub for `model.safetensors`."""
        expected = {
            "neuro-jepa-vit-b": (122054640, 96),
            "neuro-jepa-vit-l": (430759488, 192),
        }
        for name, sizes in expected.items():
            with self.subTest(name):
                encoder = _official_encoder(name)
                self.assertEqual(_sizes(encoder.state_dict()), sizes)

    def test_official_module_names(self):
        keys = set(_official_encoder("neuro-jepa-vit-b").state_dict())
        for key in (
            "patch_embed.proj.weight",
            "blocks.0.attn.qkv.weight",
            "blocks.0.attn.proj_attn_gate.weight",
            "blocks.0.attn.proj_attn_gate.bias",
            "blocks.0.mlp.fc2.bias",
            "blocks.1.mlp.gate.gate.weight",
            "blocks.1.mlp.gate.bias",
            "blocks.1.mlp.counts",
            "blocks.1.mlp.experts.15.fc1.weight",
            "blocks.1.mlp.shared_experts.fc2.weight",
            "norm.weight",
        ):
            self.assertIn(key, keys)
        self.assertNotIn("blocks.1.mlp.fc1.weight", keys)
        self.assertNotIn("blocks.0.mlp.gate.bias", keys)

    def test_official_layer_norm_epsilon(self):
        encoder = _official_encoder("neuro-jepa-vit-b")
        self.assertEqual(encoder.norm.eps, 1e-6)
        self.assertEqual(encoder.blocks[0].norm1.eps, 1e-6)

    def test_default_encoder_unchanged(self):
        """Models trained before this option must keep loading."""
        encoder = VisionTransformer3DMoE(
            img_size=(24, 24, 24), patch_size=(12, 12, 12), embed_dim=12,
            depth=1, num_heads=2,
        )
        self.assertEqual(encoder.norm.eps, 1e-5)
        self.assertFalse(
            any("proj_attn_gate" in key for key in encoder.state_dict())
        )

    def test_moe_params_as_mapping(self):
        kwargs = {
            "img_size": (24, 24, 24), "patch_size": (12, 12, 12),
            "embed_dim": 12, "depth": 2, "num_heads": 2, "use_moe": True,
        }
        params = {"dim": 12, "moe_inter_dim": 8, "moe_layer_indices": [1]}
        torch.manual_seed(0)
        from_mapping = VisionTransformer3DMoE(moe_params=params, **kwargs)
        torch.manual_seed(0)
        from_dataclass = VisionTransformer3DMoE(
            moe_params=MoEParams(**params), **kwargs
        )
        self.assertEqual(
            from_mapping.state_dict().keys(),
            from_dataclass.state_dict().keys(),
        )

    def test_attention_gate(self):
        torch.manual_seed(0)
        x = torch.rand(2, 8, 12)
        kwargs = {"D_patches": 2, "H_patches": 2, "W_patches": 2}
        reference = RoPEAttention(12, num_heads=2).eval()
        gated = RoPEAttention(12, num_heads=2, attn_gate=True).eval()
        gated.load_state_dict(reference.state_dict(), strict=False)
        torch.nn.init.zeros_(gated.proj_attn_gate.weight)

        # Fully open gates are equivalent to no gate.
        torch.nn.init.constant_(gated.proj_attn_gate.bias, 100.0)
        self.assertTrue(
            torch.allclose(gated(x, **kwargs), reference(x, **kwargs))
        )

        # Closed gates only keep the bias of the output projection.
        torch.nn.init.constant_(gated.proj_attn_gate.bias, -100.0)
        self.assertTrue(
            torch.allclose(
                gated(x, **kwargs), gated.proj.bias.expand(2, 8, 12)
            )
        )

        # A given head is only scaled by its own gate.
        torch.nn.init.constant_(gated.proj_attn_gate.bias, 100.0)
        gated.proj_attn_gate.bias.data[1] = -100.0
        gated.proj = torch.nn.Identity()
        reference.proj = torch.nn.Identity()
        out = gated(x, **kwargs)
        expected = reference(x, **kwargs)
        self.assertTrue(torch.allclose(out[..., :6], expected[..., :6]))
        self.assertTrue(torch.all(out[..., 6:].abs() < 1e-20))


class TestConfig(unittest.TestCase):
    """Configurations recorded by `ConfigMixin`."""

    def test_records_parameters(self):
        config = resnet18(n_embedding=8).get_config()
        self.assertEqual(
            config["_target_"], "nidl.backbones.volume.resnet3d.ResNet"
        )
        self.assertEqual(
            config["block"],
            {
                "_target_": "hydra.utils.get_class",
                "path": "nidl.backbones.volume.resnet3d.BasicBlock",
            },
        )
        self.assertEqual(config["layers"], [2, 2, 2, 2])
        self.assertEqual(config["n_embedding"], 8)
        encoder = instantiate(config)
        self.assertEqual(encoder.get_config(), config)
        self.assertEqual(
            {k: v.shape for k, v in encoder.state_dict().items()},
            {
                k: v.shape
                for k, v in resnet18(n_embedding=8).state_dict().items()
            },
        )

    def test_nested_estimator(self):
        config = _tiny_simclr(max_epochs=3).get_config()
        self.assertEqual(
            config["_target_"], "nidl.estimators.ssl.simclr.SimCLR"
        )
        self.assertEqual(
            config["encoder"]["_target_"],
            "nidl.backbones.volume.vit3d.VisionTransformer3D",
        )
        self.assertEqual(config["encoder"]["embed_dim"], 12)
        self.assertNotIn("max_epochs", config)
        self.assertEqual(config["proj_output_dim"], 4)
        # Default objects, such as classes of timm, are left out.
        self.assertNotIn("block_fn", config["encoder"])
        self.assertNotIn("norm_layer", config["encoder"])

    def test_trainer_parameters_left_out(self):
        config = _tiny_simclr(
            devices=1, callbacks=[], gradient_clip_val=1.0,
            enable_progress_bar=False, precision="bf16-mixed",
            max_epochs=3, save_hparams=True,
        ).get_config()
        for name in ("accelerator", "devices", "callbacks",
                     "enable_progress_bar", "gradient_clip_val",
                     "precision", "max_epochs", "save_hparams"):
            self.assertNotIn(name, config)

    def test_own_parameter_named_as_trainer_one(self):
        """`strategy` of DummyEmbeddingEstimator is not the trainer's."""
        config = DummyEmbeddingEstimator(strategy="constant").get_config()
        self.assertEqual(config["strategy"], "constant")

    def test_copy_and_pickle(self):
        model = _tiny_simclr()
        config = model.get_config()
        self.assertEqual(copy.deepcopy(model).get_config(), config)
        self.assertEqual(
            pickle.loads(pickle.dumps(model)).get_config(), config
        )
        transform = NeuroJEPAPreprocessing(input_orientation="LAS")
        self.assertEqual(
            pickle.loads(pickle.dumps(transform)).get_config(),
            transform.get_config(),
        )

    def test_unsaveable_parameter(self):
        with self.assertRaisesRegex(TypeError, "masking_fn"):
            ZNormalization(masking_fn=lambda x: x > 0).get_config()
        with self.assertRaisesRegex(TypeError, "Linear"):
            SimCLR(encoder=torch.nn.Linear(4, 4)).get_config()
        with self.assertRaisesRegex(TypeError, "no configuration"):
            get_config(torch.nn.Linear(4, 4))

    def test_classes_as_parameters(self):
        model = SimCLR(
            encoder=VisionTransformer3D,
            encoder_kwargs={"img_size": 16, "patch_size": 8, "embed_dim": 12,
                            "depth": 1, "num_heads": 2},
            proj_input_dim=12,
            optimizer=torch.optim.SGD,
        )
        config = model.get_config()
        self.assertEqual(
            config["optimizer"],
            {"_target_": "hydra.utils.get_class",
             "path": "torch.optim.sgd.SGD"},
        )
        rebuilt = instantiate(config)
        self.assertIs(rebuilt.hparams.optimizer, torch.optim.SGD)
        self.assertIsInstance(rebuilt.encoder, VisionTransformer3D)

    def test_dataclass_parameter(self):
        encoder = VisionTransformer3DMoE(
            img_size=(24, 24, 24), patch_size=(12, 12, 12), embed_dim=12,
            depth=2, num_heads=2, use_moe=True,
            moe_params=MoEParams(dim=12, moe_inter_dim=8,
                                 moe_layer_indices=[1]),
        )
        config = encoder.get_config()
        self.assertEqual(
            config["moe_params"]["_target_"],
            "nidl.backbones.volume.vit3d_moe.MoEParams",
        )
        self.assertEqual(instantiate(config).get_config(), config)

    def test_neurojepa_masks(self):
        masks = [MaskScaleConfig(spatial_scale=(0.2, 0.5), num_blocks=2)]
        config = _tiny_neurojepa(mask_scale_configs=masks).get_config()
        model = instantiate(config)
        self.assertEqual(
            [gen.cfg.num_blocks for gen in model.masker.generators],
            [2],
        )

    def test_overrides_replace_saved_values(self):
        moe_params = {"dim": 12, "n_routed_experts": 4, "moe_inter_dim": 8,
                      "moe_layer_indices": [1]}
        config = VisionTransformer3DMoE(
            img_size=(24, 24, 24), patch_size=(12, 12, 12), embed_dim=12,
            depth=2, num_heads=2, use_moe=True, moe_params=moe_params,
        ).get_config()
        override = {"dim": 12, "moe_inter_dim": 4, "moe_layer_indices": [1]}
        encoder = instantiate(config, moe_params=override)
        self.assertEqual(encoder.get_config()["moe_params"], override)

    def test_missing_parameters(self):
        config = NeuroJEPAPreprocessing(input_orientation="LAS").get_config()
        self.assertNotIn("input_orientation", config)
        with self.assertRaisesRegex(ValueError, "input_orientation"):
            instantiate(config)
        transform = instantiate(config, input_orientation="RAS")
        self.assertEqual(transform.input_orientation, "RAS")

    def test_only_allowed_classes(self):
        for config in (
            {"_target_": "os.system", "command": "ls"},
            {"_target_": "subprocess.Popen", "args": ["ls"]},
            # Imported by a nidl module, but not defined by nidl.
            {"_target_": "nidl.utils.hub.tempfile.TemporaryDirectory"},
            # Defined by nidl, but without configuration.
            {"_target_": "nidl.utils.hub.Weights", "filepath": "x"},
            {"_target_": "nidl.utils.config.instantiate"},
            {"_target_": "hydra.utils.get_class", "path": "subprocess.Popen"},
            {"_target_": "hydra.utils.get_class",
             "path": "torch.nn.modules.module.os.PathLike"},
            {"_target_": "hydra.utils.get_class", "path": "torch.nn.Linear",
             "extra": 1},
            {"_target_": "torchvision.transforms.Compose", "transforms": [],
             "other": 1},
            {"_target_": "torchvision.transforms.Normalize", "mean": [0],
             "std": [1]},
            {"_target_": "nidl.transforms.Identity",
             "p": {"_target_": "builtins.eval", "source": "1"}},
            {"_target_": 1},
            # Special keys and interpolations of Hydra.
            {"_target_": "nidl.transforms.volume.preprocessing.CropOrPad",
             "target_shape": 8, "_partial_": True},
            {"_target_": "nidl.transforms.volume.preprocessing.CropOrPad",
             "target_shape": 8, "_args_": [1]},
            {"_target_": "nidl.transforms.volume.preprocessing.CropOrPad",
             "target_shape": 8, "padding_mode": "${oc.env:HF_TOKEN}"},
        ):
            with self.subTest(config), self.assertRaises(ValueError):
                check_config(config)
            with self.subTest(config), self.assertRaises(ValueError):
                instantiate(config)

    def test_allowed_classes(self):
        for config in (
            {"_target_": "nidl.transforms.volume.preprocessing.CropOrPad",
             "target_shape": 8},
            {"_target_": "torchvision.transforms.v2.Compose",
             "transforms": []},
            {"_target_": "hydra.utils.get_class",
             "path": "torch.optim.AdamW"},
        ):
            with self.subTest(config):
                check_config(config)


class TestEstimatorHub(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def test_round_trip(self):
        model = _tiny_simclr()
        self.assertEqual(model.save_pretrained(self.directory), self.directory)
        self.assertEqual(
            sorted(p.name for p in self.directory.iterdir()),
            ["config.yaml", "model.safetensors"],
        )
        config = _read(self.directory / "config.yaml")
        self.assertEqual(sorted(config), ["estimator", "nidl_version"])
        self.assertEqual(config["estimator"], model.get_config())
        loaded = SimCLR.from_pretrained(self.directory, accelerator="cpu")
        self.assertTrue(loaded.fitted)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(loaded.state_dict()[key], value), key)

    def test_base_class_loads_saved_class(self):
        _tiny_simclr().save_pretrained(self.directory)
        self.assertIsInstance(
            BaseEstimator.from_pretrained(self.directory), SimCLR
        )

    def test_wrong_estimator(self):
        _tiny_simclr().save_pretrained(self.directory)
        with self.assertRaisesRegex(ValueError, "SimCLR"):
            YAwareContrastiveLearning.from_pretrained(self.directory)

    def test_overrides(self):
        _tiny_simclr().save_pretrained(self.directory)
        loaded = SimCLR.from_pretrained(
            self.directory, accelerator="cpu", learning_rate=1.0, devices=1
        )
        self.assertEqual(loaded.hparams.learning_rate, 1.0)
        self.assertEqual(loaded.trainer_params["devices"], 1)

    def test_trainer_parameters(self):
        """The parameters of the trainer are not saved, but can be given
        when loading."""
        _tiny_simclr(
            max_epochs=3, precision="bf16-mixed", devices=1
        ).save_pretrained(self.directory)
        loaded = SimCLR.from_pretrained(self.directory, accelerator="cpu")
        self.assertIsNone(loaded.trainer_params["max_epochs"])
        self.assertIsNone(loaded.trainer_params["precision"])
        self.assertEqual(loaded.trainer_params["devices"], "auto")
        loaded = SimCLR.from_pretrained(
            self.directory, accelerator="cpu", max_epochs=5
        )
        self.assertEqual(loaded.trainer_params["max_epochs"], 5)

    def test_not_fitted(self):
        with self.assertRaises(RuntimeError):
            _tiny_simclr(fitted=False).save_pretrained(self.directory)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_encoder_given_as_class(self):
        model = SimCLR(
            encoder=VisionTransformer3D,
            encoder_kwargs={"img_size": 16, "patch_size": 8, "embed_dim": 12,
                            "depth": 1, "num_heads": 2},
            proj_input_dim=12, accelerator="cpu",
        )
        model.fitted_ = True
        model.save_pretrained(self.directory)
        loaded = SimCLR.from_pretrained(self.directory)
        self.assertEqual(loaded.encoder.embed_dim, 12)

    def test_only_nidl_encoders(self):
        for encoder, encoder_kwargs in (
            (torch.nn.Linear(4, 4), None),
            (torch.nn.Linear, {"in_features": 4, "out_features": 4}),
        ):
            model = SimCLR(
                encoder=encoder, encoder_kwargs=encoder_kwargs,
                proj_input_dim=4,
            )
            model.fitted_ = True
            with (
                self.subTest(encoder),
                self.assertRaisesRegex(ValueError, "nidl.backbones"),
            ):
                model.save_pretrained(self.directory)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_changed_after_creation(self):
        model = _tiny_simclr()
        model.encoder.head = torch.nn.Linear(12, 3)
        with self.assertRaisesRegex(ValueError, "changed after its creation"):
            model.save_pretrained(self.directory)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_not_in_nidl_format(self):
        (self.directory / "config.yaml").write_text("model_type: x\n")
        with self.assertRaisesRegex(ValueError, "nidl format"):
            SimCLR.from_pretrained(self.directory)
        with self.assertRaises(FileNotFoundError):
            Transform.from_pretrained(self.directory)

    def test_unsafe_configuration(self):
        model = _tiny_simclr()
        model.save_pretrained(self.directory)
        config = _read(self.directory / "config.yaml")
        config["estimator"]["encoder"] = {"_target_": "os.system"}
        _write_config(config, self.directory / "config.yaml")
        with (
            mock.patch("os.system") as system,
            self.assertRaisesRegex(ValueError, "os.system"),
        ):
            SimCLR.from_pretrained(self.directory)
        system.assert_not_called()

    def test_version(self):
        _tiny_simclr().save_pretrained(self.directory)
        config = _read(self.directory / "config.yaml")
        config["nidl_version"] = "99.0"
        _write_config(config, self.directory / "config.yaml")
        with self.assertWarnsRegex(UserWarning, "99.0"):
            SimCLR.from_pretrained(self.directory)

    def test_strict(self):
        model = _tiny_simclr()
        model.save_pretrained(self.directory)
        state_dict = model.state_dict()
        del state_dict["encoder.norm.weight"]
        save_file(state_dict, str(self.directory / "model.safetensors"))
        with self.assertRaisesRegex(RuntimeError, "encoder.norm.weight"):
            SimCLR.from_pretrained(self.directory)
        with self.assertWarnsRegex(UserWarning, "encoder.norm.weight"):
            SimCLR.from_pretrained(self.directory, strict=False)

    def test_hub_download(self):
        _tiny_simclr().save_pretrained(self.directory)
        data_dir = self.directory / "cache"

        def download(repo_id, filename, **kwargs):
            self.assertEqual(repo_id, "someone/model")
            self.assertEqual(
                kwargs["local_dir"], data_dir / "someone_model"
            )
            return str(self.directory / filename)

        with mock.patch(
            "huggingface_hub.hf_hub_download", side_effect=download
        ) as hf_hub_download:
            SimCLR.from_pretrained(
                "someone/model", revision="abc", data_dir=data_dir
            )
        self.assertEqual(
            [c.kwargs["filename"] for c in hf_hub_download.call_args_list],
            ["config.yaml", "model.safetensors"],
        )
        self.assertEqual(
            {c.kwargs["revision"] for c in hf_hub_download.call_args_list},
            {"abc"},
        )

    def test_gated_repository(self):
        response = mock.Mock(status_code=403, headers={})
        error = GatedRepoError("gated", response=response)
        with (
            mock.patch("huggingface_hub.hf_hub_download", side_effect=error),
            self.assertRaisesRegex(PermissionError, "someone/gated"),
        ):
            NeuroJEPA.from_pretrained(
                "someone/gated", data_dir=self.directory
            )

    def test_push_to_hub(self):
        api = mock.MagicMock()

        def upload_folder(folder_path, **kwargs):
            self.assertEqual(
                sorted(p.name for p in Path(folder_path).iterdir()),
                ["config.yaml", "model.safetensors"],
            )
            self.assertEqual(kwargs["repo_id"], "someone/model")
            return "commit"

        api.upload_folder.side_effect = upload_folder
        with mock.patch("huggingface_hub.HfApi", return_value=api):
            commit = _tiny_simclr().push_to_hub("someone/model", private=True)
        self.assertEqual(commit, "commit")
        api.create_repo.assert_called_once_with(
            "someone/model", private=True, exist_ok=True
        )
        api.upload_folder.assert_called_once()

    def test_fine_tune_after_from_pretrained(self):
        _tiny_simclr().save_pretrained(self.directory)
        model = SimCLR.from_pretrained(
            self.directory, accelerator="cpu", devices=1, max_epochs=1,
            logger=False, enable_progress_bar=False,
            enable_checkpointing=False, default_root_dir=self.directory,
        )
        before = {k: v.clone() for k, v in model.state_dict().items()}
        x = torch.rand(4, 1, 16, 16, 16)
        views = [[a, a + 0.01 * torch.randn_like(a)] for a in x]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(DataLoader(views, batch_size=2))
        self.assertTrue(model.fitted)
        self.assertFalse(
            torch.equal(
                before["encoder.norm.weight"],
                model.state_dict()["encoder.norm.weight"],
            )
        )


class TestExternalWeights(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def test_contract(self):
        with self.assertRaises(TypeError):
            ExternalWeights()

    def test_neurojepa(self):
        tensor = torch.zeros(1)
        converted = NeuroJEPAWeights().convert(
            {"module.backbone.norm.weight": tensor, "other": tensor}
        )
        self.assertEqual(
            set(converted),
            {
                "context_encoder.vit.norm.weight",
                "target_encoder.vit.norm.weight",
                "other",
            },
        )

    def test_get_converter(self):
        self.assertIsInstance(
            get_converter("NeuroJEPAWeights"), NeuroJEPAWeights
        )
        for name in ("ExternalWeights", "torch", "Mapping", "_x", None):
            with self.subTest(name), self.assertRaises(ValueError):
                get_converter(name)

    def _authors_weights(self, state_dict=None):
        """Publish the weights of a tiny NeuroJEPA in the authors' format."""
        model = _tiny_neurojepa()
        if state_dict is None:
            state_dict = {
                f"module.backbone.{k}": v
                for k, v in model.context_encoder.vit.state_dict().items()
            }
        authors = self.directory / "authors"
        authors.mkdir(exist_ok=True)
        save_file(state_dict, str(authors / "model.safetensors"))
        return model

    def _load_external(self, **weights):
        """Load the tiny NeuroJEPA from its nidl configuration."""
        _write_config(
            {
                "nidl_version": "0.0.1",
                "estimator": _tiny_neurojepa().get_config(),
                "weights": {
                    "repo_id": AUTHORS,
                    "revision": "0" * 40,
                    "converter": "NeuroJEPAWeights",
                    "missing_ok": ["predictor."],
                    **weights,
                },
            },
            self.directory / "config.yaml",
        )

        def download(repo_id, filename, revision, **kwargs):
            self.assertEqual(repo_id, AUTHORS)
            self.assertEqual(revision, "0" * 40)
            return str(self.directory / "authors" / filename)

        with mock.patch(
            "huggingface_hub.hf_hub_download", side_effect=download
        ):
            return NeuroJEPA.from_pretrained(
                self.directory, accelerator="cpu", data_dir=self.directory
            )

    def test_external_weights(self):
        expected = self._authors_weights().context_encoder.vit.state_dict()
        loaded = self._load_external()
        self.assertTrue(loaded.fitted)
        for encoder in (loaded.context_encoder, loaded.target_encoder):
            state_dict = encoder.vit.state_dict()
            self.assertEqual(set(state_dict), set(expected))
            for key, value in expected.items():
                self.assertTrue(torch.equal(state_dict[key], value), key)
        target_parameters = loaded.target_encoder.parameters()
        self.assertTrue(all(not p.requires_grad for p in target_parameters))

    def test_external_weights_strict(self):
        state_dict = {
            f"module.backbone.{k}": v
            for k, v in self._authors_weights()
            .context_encoder.vit.state_dict()
            .items()
        }
        del state_dict["module.backbone.norm.weight"]
        self._authors_weights(state_dict)
        with self.assertRaisesRegex(RuntimeError, "norm.weight"):
            self._load_external()

    def test_external_weights_need_revision_and_converter(self):
        self._authors_weights()
        with self.assertRaisesRegex(ValueError, "revision"):
            self._load_external(revision=None)
        for converter in ("ExternalWeights", "torch", "Mapping"):
            with self.subTest(converter), self.assertRaises(ValueError):
                self._load_external(converter=converter)

    def test_ready_for_transform(self):
        self._authors_weights()
        model = self._load_external()
        x = torch.rand(4, *TINY_SHAPE)
        model.trainer_params_.update(
            enable_progress_bar=False, devices=1, logger=False
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features = model.transform(DataLoader(x, batch_size=2))
        self.assertEqual(features.shape, (4, 12))

    def test_save_after_loading(self):
        """All the weights are saved, without the `weights` entry."""
        self._authors_weights()
        model = self._load_external()
        output = self.directory / "saved"
        model.save_pretrained(output)
        self.assertNotIn("weights", _read(output / "config.yaml"))
        reloaded = NeuroJEPA.from_pretrained(output, accelerator="cpu")
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(reloaded.state_dict()[key], value))


class TestTransformHub(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.x = np.random.default_rng(0).random((1, 9, 9, 9))

    def test_round_trip(self):
        transform = CropOrPad(target_shape=8, padding_mode="edge")
        transform.save_pretrained(self.directory)
        self.assertEqual(
            [p.name for p in self.directory.iterdir()],
            ["preprocessor_config.yaml"],
        )
        config = _read(self.directory / "preprocessor_config.yaml")
        self.assertEqual(config["transform"], transform.get_config())
        for cls in (Transform, CropOrPad):
            with self.subTest(cls):
                loaded = cls.from_pretrained(self.directory)
                self.assertIsInstance(loaded, CropOrPad)
                self.assertTrue(
                    np.array_equal(loaded(self.x), transform(self.x))
                )
        with self.assertRaisesRegex(ValueError, "CropOrPad"):
            ZNormalization.from_pretrained(self.directory)

    def test_overrides(self):
        BrainIACPreprocessing().save_pretrained(self.directory)
        crop = ((1, 8), (1, 8), (1, 8))
        loaded = Transform.from_pretrained(self.directory, crop=crop)
        self.assertEqual(loaded.get_config()["crop"], [[1, 8]] * 3)

    def test_data_parameters_not_saved(self):
        for cls in (NeuroJEPAPreprocessing, NeuroSimCLRPreprocessing):
            with self.subTest(cls):
                cls(input_orientation="LAS").save_pretrained(self.directory)
                config = _read(self.directory / "preprocessor_config.yaml")
                self.assertNotIn("input_orientation", config["transform"])
                with self.assertRaisesRegex(ValueError, "input_orientation"):
                    Transform.from_pretrained(self.directory)
                loaded = Transform.from_pretrained(
                    self.directory, input_orientation="RAS"
                )
                self.assertIsInstance(loaded, cls)
                self.assertEqual(loaded.input_orientation, "RAS")

    def test_torchvision_containers(self):
        x = torch.from_numpy(self.x)
        for module in (transforms_v1, transforms_v2):
            for container in (
                module.Compose([CropOrPad(8), ZNormalization()]),
                module.RandomApply([CropOrPad(8)], p=1.0),
                module.RandomChoice([CropOrPad(8)]),
                module.RandomOrder([CropOrPad(8)]),
            ):
                with self.subTest(container):
                    save_pretrained(container, self.directory)
                    loaded = Transform.from_pretrained(self.directory)
                    self.assertIs(type(loaded), type(container))
                    self.assertTrue(torch.equal(loaded(x), container(x)))
                    with self.assertRaisesRegex(ValueError, "Compose|Random"):
                        CropOrPad.from_pretrained(self.directory)

    def test_torchvision_transforms_not_saved(self):
        for container in (
            transforms_v1.Compose([transforms_v1.Normalize([0], [1])]),
            transforms_v2.Compose([lambda x: x]),
        ):
            with self.subTest(container), self.assertRaises(TypeError):
                save_pretrained(container, self.directory)
        with self.assertRaisesRegex(TypeError, "cannot be saved"):
            save_pretrained(transforms_v1.Normalize([0], [1]), self.directory)

    def test_same_repository_as_estimator(self):
        model = _tiny_simclr()
        model.save_pretrained(self.directory)
        ZNormalization().save_pretrained(self.directory)
        self.assertEqual(
            sorted(p.name for p in self.directory.iterdir()),
            ["config.yaml", "model.safetensors", "preprocessor_config.yaml"],
        )
        self.assertIsInstance(
            Transform.from_pretrained(self.directory), ZNormalization
        )
        self.assertIsInstance(SimCLR.from_pretrained(self.directory), SimCLR)

    def test_push_to_hub(self):
        api = mock.MagicMock()

        def upload_folder(folder_path, **kwargs):
            self.assertEqual(
                [p.name for p in Path(folder_path).iterdir()],
                ["preprocessor_config.yaml"],
            )

        api.upload_folder.side_effect = upload_folder
        with mock.patch("huggingface_hub.HfApi", return_value=api):
            ZNormalization().push_to_hub("someone/model")
            push_to_hub(
                transforms_v1.Compose([ZNormalization()]), "someone/model"
            )
        self.assertEqual(api.upload_folder.call_count, 2)


class TestPublishedArchitectures(unittest.TestCase):
    """Architectures of the weights published in the nidl format."""

    def _build(self, name):
        return PUBLISHED[name]["estimator"]()

    def test_yaware(self):
        state_dict = self._build("yaware-densenet121-bhb10k").state_dict()
        self.assertEqual(len(state_dict), 724)
        self.assertEqual(
            sum(v.numel() for v in state_dict.values()), 11912760
        )
        self.assertEqual(
            state_dict["projection_head.layers.0.weight"].shape, (512, 1024)
        )

    def test_neuro_simclr(self):
        model = self._build("neuro-simclr-resnet18")
        state_dict = model.state_dict()
        self.assertEqual(
            sum(v.numel() for v in state_dict.values()), 33465428
        )
        self.assertEqual(len(state_dict), 125)
        self.assertEqual(model.encoder.conv1.stride, (1, 1, 1))
        self.assertIn("encoder.layer2.0.downsample.0.bias", state_dict)
        self.assertNotIn("projection_head.layers.0.bias", state_dict)

    def test_brainiac(self):
        model = self._build("brainiac-vit-b").eval()
        state_dict = model.state_dict()
        self.assertEqual(
            sum(v.numel() for v in state_dict.values()), 90514690
        )
        self.assertEqual(len(state_dict), 149)
        self.assertNotIn("encoder.cls_token", state_dict)
        self.assertEqual(state_dict["encoder.pos_embed"].shape, (1, 216, 768))
        x = torch.rand(2, 1, 96, 96, 96)
        with torch.no_grad():
            features = model.transform_step(x, 0)
            tokens = model.encoder.forward_features(x)
        self.assertTrue(torch.equal(features, tokens.mean(dim=1)))
        self.assertIsInstance(model.encoder.norm, torch.nn.LayerNorm)

    def test_documentation_registry(self):
        """The models advertised in the documentation are the published
        ones, with their preprocessing."""
        registry = Path(__file__).parents[2] / "doc" / "pretrained_models.json"
        if not registry.is_file():
            self.skipTest("documentation not available")
        models = json.loads(registry.read_text())["models"]
        published = PUBLISHED
        self.assertEqual(
            {m["repo_id"] for m in models},
            {f"nidl-neurospin/{name}" for name in published},
        )
        for model in models:
            with self.subTest(model["repo_id"]):
                info = published[model["repo_id"].split("/")[1]]
                preprocessing = info["preprocessing"]()
                self.assertEqual(
                    model["preprocessing"], type(preprocessing).__name__
                )
                self.assertEqual(model["gated"], info["gated"])
                for key in ("subjects", "volumes", "modality"):
                    self.assertIn(key, model)
                # The parameters given for OpenBHB complete the saved ones.
                instantiate(
                    preprocessing.get_config(),
                    **_tuples(model["openbhb"]["kwargs"]),
                )

    def test_default_models_unchanged(self):
        self.assertIn("embedding.weight", densenet121().state_dict())
        state_dict = resnet18().state_dict()
        self.assertIn("embedding.weight", state_dict)
        self.assertNotIn("layer2.0.downsample.0.bias", state_dict)
        self.assertEqual(resnet18().conv1.stride, (2, 2, 2))
        keys = SimCLRProjectionHead(8, 8, 4).state_dict()
        self.assertEqual(
            set(keys),
            {f"layers.{i}.{p}" for i in (0, 2) for p in ("weight", "bias")},
        )


def test_config_mixin_is_shared():
    """Backbones, estimators and transforms record their parameters."""
    for cls in (VisionTransformer3D, SimCLR, Transform, CropForeground):
        assert issubclass(cls, ConfigMixin)
    assert Reorient("LAS").get_config()["input_orientation"] == "LAS"


class TestReorient(unittest.TestCase):
    def test_flip_and_permute(self):
        ras = np.random.default_rng(0).random((1, 4, 5, 6))
        self.assertTrue(
            np.array_equal(Reorient("LAS")(ras[:, ::-1]), ras)
        )
        asr = ras[0].transpose(1, 2, 0)
        self.assertTrue(np.array_equal(Reorient("ASR")(asr), ras[0]))
        self.assertTrue(
            np.array_equal(Reorient("RAS", "LAS")(ras), ras[:, ::-1])
        )

    def test_tensor(self):
        x = torch.rand(1, 4, 5, 6)
        output = Reorient("LAS")(x)
        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(torch.equal(output, x.flip(1)))

    def test_invalid_orientation(self):
        for orientation in ("XYZ", "RR", "RLA", ""):
            with self.subTest(orientation), self.assertRaises(ValueError):
                Reorient(orientation)


class TestCropForeground(unittest.TestCase):
    def test_crop(self):
        x = np.zeros((1, 30, 30, 30))
        x[0, 10:12, 1:3, 25:27] = 1.0
        cropped = CropForeground(margin=4)(x)
        self.assertEqual(cropped.shape, (1, 10, 7, 9))
        self.assertEqual(cropped.sum(), 8)
        self.assertEqual(CropForeground()(x[0]).shape, (2, 2, 2))

    def test_any_channel_and_tensor(self):
        x = torch.zeros(2, 10, 10, 10)
        x[0, 2, 2, 2] = 1.0
        x[1, 5, 6, 7] = 1.0
        self.assertEqual(CropForeground()(x).shape, (2, 4, 5, 6))
        self.assertEqual(
            CropForeground(threshold=1.0)(x + 0.5).shape, (2, 4, 5, 6)
        )

    def test_no_foreground(self):
        x = np.zeros((1, 4, 4, 4))
        np.testing.assert_array_equal(CropForeground(margin=4)(x), x)
        with self.assertRaises(ValueError):
            CropForeground(margin=-1)


class TestSimCLRPreprocessing(unittest.TestCase):
    def test_brainiac(self):
        volume = np.zeros((1, 182, 218, 182), dtype=np.float32)
        volume[0, 40:140, 40:170, 40:140] = 100 + np.random.default_rng(
            0
        ).normal(0, 10, (100, 130, 100))
        transform = BrainIACPreprocessing(crop=((6, 176), (7, 213), (3, 165)))
        output = transform(volume)
        self.assertEqual(output.shape, (1, 96, 96, 96))
        self.assertEqual(output.dtype, torch.float32)
        foreground = output[output != 0]
        self.assertAlmostEqual(foreground.mean().item(), 0.0, places=4)
        self.assertAlmostEqual(foreground.std().item(), 1.0, places=2)

    def test_neuro_simclr_geometry(self):
        """A voxel of known world coordinates lands at the expected index."""
        volume = np.zeros((1, 182, 218, 182), dtype=np.float32)
        # In LAS, x = 90 - i, y = -126 + j, z = -72 + k.
        volume[0, 90, 126, 72] = 3.0  # the origin
        volume[0, 91, 127, 73] = 1.0
        volume[0, 89, 125, 71] = 1.0
        transform = NeuroSimCLRPreprocessing(
            "LAS", padding=((5, 6), (6, 5), (6, 5))
        )
        output = transform(volume)
        self.assertEqual(output.shape, (1, 150, 192, 192))
        # The origin is at (96, 132, 78) on the template, then (z, y, x) minus
        # the crop start.
        index = output[0].argmax().item()
        expected = (78 - 10) * 192 * 192 + (132 - 19) * 192 + (96 - 1)
        self.assertEqual(index, expected)

    def test_neuro_simclr_needs_template_grid(self):
        with self.assertRaisesRegex(ValueError, "grid"):
            NeuroSimCLRPreprocessing("RAS")(np.ones((1, 182, 218, 182)))

    def test_standardization_without_foreground(self):
        with self.assertRaisesRegex(ValueError, "foreground"):
            BrainIACPreprocessing()(np.zeros((1, 32, 32, 32)))


class TestNeuroJEPAPreprocessing(unittest.TestCase):
    def _brain(self, shape=(182, 218, 182), seed=0):
        rng = np.random.default_rng(seed)
        grid = np.indices(shape, dtype=np.float32)
        center = np.array(shape).reshape(3, 1, 1, 1) / 2
        distance = (((grid - center) / (0.4 * center)) ** 2).sum(0)
        return (distance < 1) * (100 + rng.normal(0, 20, shape))[None]

    def test_output(self):
        output = NeuroJEPAPreprocessing("RAS")(self._brain())
        self.assertEqual(output.shape, (1, 96, 108, 96))
        self.assertEqual(output.dtype, torch.float32)
        self.assertGreaterEqual(output.min().item(), 0.0)
        self.assertLessEqual(output.max().item(), 1.0)

    def test_inputs_not_modified_and_equivalent(self):
        volume = self._brain().astype(np.float32)
        volume[0, 0, 0, 0] = np.nan
        array, tensor = volume.copy(), torch.from_numpy(volume.copy())
        transform = NeuroJEPAPreprocessing("RAS")
        expected = transform(array)
        self.assertTrue(np.array_equal(array, volume, equal_nan=True))
        self.assertTrue(torch.equal(expected, transform(tensor)))
        self.assertTrue(torch.equal(expected, transform(array[0])))
        self.assertTrue(torch.isfinite(expected).all())

    def test_orientation(self):
        """The same brain given in RAS or LAS gives the same output."""
        ras = self._brain()
        ras[0, 110:, :, :] *= 2  # asymmetric
        las = ras[:, ::-1].copy()
        expected = NeuroJEPAPreprocessing("RAS")(ras)
        self.assertTrue(
            torch.equal(expected, NeuroJEPAPreprocessing("LAS")(las))
        )
        self.assertFalse(
            torch.equal(expected, NeuroJEPAPreprocessing("RAS")(las))
        )

    def test_percentiles_before_cropping(self):
        """Voxels cropped later on must still set the rescaling range: this
        is the order of the official code, not the one of the model card."""
        volume = self._brain()
        self.assertGreater(
            NeuroJEPAPreprocessing("RAS")(volume).max().item(), 0.9
        )
        volume[0, [0, -1]] = 1e5  # 1.1% of the voxels, in the cropped slices
        self.assertLess(
            NeuroJEPAPreprocessing("RAS")(volume).max().item(), 0.1
        )

    def test_constant_volume(self):
        with self.assertRaisesRegex(ValueError, "foreground"):
            NeuroJEPAPreprocessing("RAS")(np.ones((1, 182, 218, 182)))

    def test_matches_monai(self):
        """Cross-check against the transforms of the official pipeline."""
        try:
            from monai import transforms as mt
        except ImportError:
            self.skipTest("monai is not installed")
        volume = self._brain().astype(np.float32)
        pipeline = mt.Compose([
            mt.ScaleIntensityRangePercentiles(
                lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True
            ),
            mt.ResizeWithPadOrCrop([180, 216, 180], mode="edge"),
            mt.CropForeground(
                select_fn=lambda x: x > 0.0, margin=4, allow_smaller=True
            ),
            mt.Resize([100, 120, 100]),
            mt.ResizeWithPadOrCrop(
                [96, 108, 96], mode="constant", value=0
            ),
            mt.CenterSpatialCrop([96, 108, 96]),
        ])
        for offset in (0.0, -3.7):  # the lower percentile may be negative
            shifted = volume + np.float32(offset)
            expected = torch.as_tensor(pipeline(torch.from_numpy(shifted)))
            output = NeuroJEPAPreprocessing("RAS")(shifted)
            self.assertTrue(torch.equal(output, expected))

    def test_brainiac_matches_monai(self):
        try:
            from monai import transforms as mt
        except ImportError:
            self.skipTest("monai is not installed")
        volume = self._brain().astype(np.float32)
        pipeline = mt.Compose([
            mt.Resize([96, 96, 96], mode="trilinear"),
            mt.NormalizeIntensity(nonzero=True, channel_wise=True),
        ])
        expected = torch.as_tensor(pipeline(torch.from_numpy(volume)))
        output = BrainIACPreprocessing()(volume)
        self.assertTrue(torch.allclose(output, expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
