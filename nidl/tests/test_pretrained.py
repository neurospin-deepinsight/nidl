##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import importlib.util
import json
import re
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from huggingface_hub.utils import GatedRepoError
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from nidl.backbones.volume.densenet3d import densenet121
from nidl.backbones.volume.resnet3d import resnet18
from nidl.backbones.volume.vit3d import VisionTransformer3D
from nidl.backbones.volume.vit3d_moe import (
    MoEParams,
    RoPEAttention,
    VisionTransformer3DMoE,
)
from nidl.estimators.external import ExternalWeights, NeuroJEPAWeights
from nidl.estimators.ssl import NeuroJEPA, SimCLR, YAwareContrastiveLearning
from nidl.estimators.ssl.utils.projection_heads import SimCLRProjectionHead
from nidl.transforms.volume import preprocessing
from nidl.transforms.volume.preprocessing import (
    BrainIACPreprocessing,
    CropForeground,
    NeuroJEPAPreprocessing,
    NeuroSimCLRPreprocessing,
    Reorient,
)
from nidl.utils.weights import Weights

TINY_SHAPE = (1, 24, 24, 24)
TOOLS = Path(__file__).parents[2] / "tools" / "convert_pretrained_weights.py"
AUTHORS = "authors/tiny-jepa"


def _load_tool():
    spec = importlib.util.spec_from_file_location("convert", TOOLS)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def _official_encoder(name):
    info = _load_tool().EXTERNAL_MODELS[name]
    config = {"estimator": info["estimator"], **info["config"]}
    with mock.patch.object(torch.nn.Module, "load_state_dict"):
        model = NeuroJEPA.from_config(config, accelerator="cpu")
    return model.context_encoder.vit


def _tiny_neurojepa_config(**weights):
    """A tiny NeuroJEPA whose weights are published by their authors."""
    return {
        "estimator": "NeuroJEPA",
        "estimator_kwargs": {
            "use_moe": True,
            "predictor_embed_dim": 8,
            "predictor_depth": 1,
            "predictor_num_heads": 2,
        },
        "encoder": {
            "name": "VisionTransformer3DMoE",
            "kwargs": {
                "img_size": list(TINY_SHAPE[1:]),
                "patch_size": [12, 12, 12],
                "embed_dim": 12,
                "depth": 2,
                "num_heads": 2,
                "use_moe": True,
                "moe_params": {
                    "dim": 12,
                    "n_routed_experts": 4,
                    "n_activated_experts": 2,
                    "moe_inter_dim": 8,
                    "moe_layer_indices": [1],
                },
                "attn_gate": True,
                "norm_eps": 1e-6,
            },
        },
        "input": {"shape": list(TINY_SHAPE), "range": [0.0, 1.0]},
        "preprocessing": {
            "transform": "NeuroJEPAPreprocessing", "kwargs": {}
        },
        "weights": {
            "repo_id": AUTHORS,
            "revision": "0" * 40,
            "converter": "NeuroJEPAWeights",
            "missing_ok": ["predictor."],
            **weights,
        },
    }


def _tuples(value):
    if isinstance(value, dict):
        return {k: _tuples(v) for k, v in value.items()}
    if isinstance(value, list):
        return tuple(_tuples(v) for v in value)
    return value


def _sizes(state_dict):
    floats = sum(
        v.numel() for v in state_dict.values() if v.is_floating_point()
    )
    integers = sum(
        v.numel() for v in state_dict.values() if not v.is_floating_point()
    )
    return floats, integers


def _tiny_vit_config():
    return {
        "estimator": "SimCLR",
        "estimator_kwargs": {
            "proj_input_dim": 12,
            "proj_hidden_dim": 8,
            "proj_output_dim": 4,
        },
        "encoder": {
            "name": "VisionTransformer3D",
            "kwargs": {
                "img_size": 16,
                "patch_size": 8,
                "embed_dim": 12,
                "depth": 1,
                "num_heads": 2,
            },
        },
        "input": {"shape": [1, 16, 16, 16], "range": [0.0, 1.0]},
        "preprocessing": {
            "transform": "RobustRescaling",
            "kwargs": {"percentiles": [1, 99]},
        },
    }


@unittest.skipUnless(TOOLS.is_file(), "conversion tool not available")
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


class TestExternalWeights(unittest.TestCase):
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


class TestNidlHubMixin(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def _model(self, **kwargs):
        return SimCLR.from_config(
            _tiny_vit_config(), accelerator="cpu", **kwargs
        )

    def _authors_weights(self, state_dict=None):
        """Publish the weights of a tiny NeuroJEPA in the authors' format."""
        model = NeuroJEPA.from_config(_tiny_neurojepa_config())
        if state_dict is None:
            state_dict = {
                f"module.backbone.{k}": v
                for k, v in model.context_encoder.vit.state_dict().items()
            }
        authors = self.directory / "authors"
        authors.mkdir(exist_ok=True)
        save_file(state_dict, str(authors / "model.safetensors"))
        return model

    def _load_external(self, config=None, **kwargs):
        """Load the tiny NeuroJEPA from its nidl configuration."""
        (self.directory / "config.json").write_text(
            json.dumps(config or _tiny_neurojepa_config())
        )

        def download(_, repo_id, filename, **__):
            self.assertEqual(repo_id, AUTHORS)
            return self.directory / "authors" / filename

        with mock.patch.object(
            Weights, "hf_download", side_effect=download
        ) as hf_download:
            model = NeuroJEPA.from_pretrained(
                self.directory, accelerator="cpu", **kwargs
            )
        self.assertEqual(
            hf_download.call_args.kwargs["hf_revision"], "0" * 40
        )
        return model

    def test_round_trip(self):
        model = self._model()
        model.save_pretrained(self.directory)
        loaded = SimCLR.from_pretrained(self.directory, accelerator="cpu")
        self.assertTrue(loaded.fitted)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(loaded.state_dict()[key], value), key)
        config = json.loads((self.directory / "config.json").read_text())
        self.assertEqual(config["encoder"], _tiny_vit_config()["encoder"])
        self.assertEqual(config["input"], _tiny_vit_config()["input"])
        self.assertNotIn("accelerator", config["estimator_kwargs"])
        self.assertEqual(config["estimator_kwargs"]["proj_output_dim"], 4)

    def test_estimator_overrides(self):
        self._model().save_pretrained(self.directory)
        loaded = SimCLR.from_pretrained(
            self.directory, accelerator="cpu", learning_rate=1.0
        )
        self.assertEqual(loaded.learning_rate, 1.0)

    def test_fine_tune_after_from_pretrained(self):
        self._model().save_pretrained(self.directory)
        model = SimCLR.from_pretrained(
            self.directory, accelerator="cpu", devices=1, max_epochs=1,
            logger=False, enable_progress_bar=False,
            enable_checkpointing=False,
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

    def test_encoder_given_as_class(self):
        model = SimCLR(
            encoder=VisionTransformer3D,
            encoder_kwargs={"img_size": 16, "patch_size": 8, "embed_dim": 12,
                            "depth": 1, "num_heads": 2},
            proj_input_dim=12, accelerator="cpu",
        )
        model.save_pretrained(self.directory)
        config = json.loads((self.directory / "config.json").read_text())
        self.assertEqual(config["encoder"]["name"], "VisionTransformer3D")
        self.assertEqual(config["encoder"]["kwargs"]["embed_dim"], 12)
        self.assertNotIn("input", config)
        SimCLR.from_pretrained(self.directory, accelerator="cpu")

    def test_encoder_instance_needs_its_config(self):
        model = SimCLR(encoder=resnet18(), proj_input_dim=512)
        with self.assertRaisesRegex(ValueError, "encoder"):
            model.save_pretrained(self.directory)
        model.save_pretrained(
            self.directory, encoder={"name": "resnet18", "kwargs": {}}
        )

    def test_wrong_estimator(self):
        self._model().save_pretrained(self.directory)
        with self.assertRaisesRegex(ValueError, "SimCLR"):
            YAwareContrastiveLearning.from_pretrained(self.directory)

    def test_only_nidl_objects(self):
        for name in ("system", "os", "_VisionTransformer3D", "Mapping", 1):
            with self.subTest(name):
                config = _tiny_vit_config()
                config["encoder"]["name"] = name
                with self.assertRaisesRegex(ValueError, "Unknown encoder"):
                    SimCLR.from_config(config)
        config = _tiny_vit_config()
        config["preprocessing"]["transform"] = "torch"
        with self.assertRaisesRegex(ValueError, "Unknown preprocessing"):
            SimCLR.from_config(config).get_preprocessing()

    def test_preprocessing(self):
        model = self._model()
        transform = model.get_preprocessing()
        self.assertEqual(transform.percentiles, (1, 99))
        with self.assertRaisesRegex(ValueError, "percentiles"):
            model.get_preprocessing(percentiles=(0, 100))
        self.assertIsNone(SimCLR(encoder=resnet18()).get_preprocessing())

    def test_preprocessing_needs_data_parameters(self):
        config = _tiny_vit_config()
        config["preprocessing"] = {
            "transform": "NeuroJEPAPreprocessing", "kwargs": {}
        }
        model = SimCLR.from_config(config)
        with self.assertRaisesRegex(TypeError, "input_orientation"):
            model.get_preprocessing()
        self.assertIsInstance(
            model.get_preprocessing(input_orientation="LAS"),
            NeuroJEPAPreprocessing,
        )

    def test_input_checks(self):
        model = self._model()
        model.check_pretrained_input(torch.rand(2, 1, 16, 16, 16))
        with self.assertRaisesRegex(ValueError, "shape"):
            model.transform_step(torch.rand(2, 1, 8, 16, 16), 0)
        with self.assertRaisesRegex(ValueError, "values in"):
            model.transform_step(2 * torch.ones(2, 1, 16, 16, 16), 0)
        x = torch.rand(2, 1, 16, 16, 16)
        x[0, 0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            model.transform_step(x, 0)

    def test_no_check_without_pretrained(self):
        model = SimCLR(encoder=resnet18(), proj_input_dim=512)
        model.check_pretrained_input(torch.rand(1, 1, 3, 3, 3) * 10)

    def test_strict(self):
        model = self._model()
        model.save_pretrained(self.directory)
        state_dict = model.state_dict()
        del state_dict["encoder.norm.weight"]
        save_file(state_dict, str(self.directory / "model.safetensors"))
        with self.assertRaisesRegex(RuntimeError, "encoder.norm.weight"):
            SimCLR.from_pretrained(self.directory)
        with self.assertWarnsRegex(UserWarning, "encoder.norm.weight"):
            SimCLR.from_pretrained(self.directory, strict=False)

    def test_not_in_nidl_format(self):
        (self.directory / "config.json").write_text('{"model_type": "x"}')
        with self.assertRaisesRegex(ValueError, "nidl format"):
            SimCLR.from_pretrained(self.directory)

    def test_hub_download(self):
        self._model().save_pretrained(self.directory)
        with mock.patch.object(
            Weights,
            "hf_download",
            side_effect=lambda _, __, name, **___: self.directory / name,
        ) as download:
            SimCLR.from_pretrained("someone/model", accelerator="cpu")
            SimCLR.from_pretrained(
                "someone/model", revision="abc", accelerator="cpu"
            )
        revisions = [c.kwargs["hf_revision"] for c in download.call_args_list]
        self.assertEqual(revisions, [None, None, "abc", "abc"])
        self.assertEqual(
            {c.args[1] for c in download.call_args_list}, {"someone/model"}
        )

    def test_gated_repository(self):
        response = mock.Mock(status_code=403, headers={})
        error = GatedRepoError("gated", response=response)
        with (
            mock.patch.object(Weights, "hf_download", side_effect=error),
            self.assertRaisesRegex(PermissionError, "someone/gated"),
        ):
            NeuroJEPA.from_pretrained("someone/gated")

    def test_push_to_hub(self):
        api = mock.MagicMock()

        def upload_folder(folder_path, **kwargs):
            self.assertEqual(
                sorted(p.name for p in Path(folder_path).iterdir()),
                ["config.json", "model.safetensors"],
            )
            return "commit"

        api.upload_folder.side_effect = upload_folder
        with mock.patch("huggingface_hub.HfApi", return_value=api):
            commit = self._model().push_to_hub("someone/model", private=True)
        self.assertEqual(commit, "commit")
        api.create_repo.assert_called_once_with(
            "someone/model", private=True, exist_ok=True
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
        self.assertIsInstance(
            loaded.get_preprocessing(input_orientation="LAS"),
            NeuroJEPAPreprocessing,
        )

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
            self._load_external(_tiny_neurojepa_config(revision=None))
        for converter in ("ExternalWeights", "torch", "Mapping"):
            with self.subTest(converter), self.assertRaises(ValueError):
                self._load_external(
                    _tiny_neurojepa_config(converter=converter)
                )

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

    def test_save_keeps_pretrained_config(self):
        self._authors_weights()
        model = self._load_external()
        output = self.directory / "saved"
        model.save_pretrained(output)
        config = json.loads((output / "config.json").read_text())
        self.assertEqual(config["estimator"], "NeuroJEPA")
        self.assertNotIn("weights", config)  # all the weights are saved
        self.assertEqual(
            config["preprocessing"]["transform"], "NeuroJEPAPreprocessing"
        )
        reloaded = NeuroJEPA.from_pretrained(output, accelerator="cpu")
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(reloaded.state_dict()[key], value))


@unittest.skipUnless(TOOLS.is_file(), "conversion tool not available")
class TestConvertedArchitectures(unittest.TestCase):
    """Architectures of the weights converted to the nidl format."""

    @classmethod
    def setUpClass(cls):
        cls.tool = _load_tool()

    def _build(self, name):
        info = self.tool.MODELS[name]
        config = {"estimator": info["estimator"].__name__, **info["config"]}
        return info["estimator"].from_config(config, accelerator="cpu")

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

    def test_original_names(self):
        """Original names of each checkpoint, renamed by the tool."""
        expected = {
            "yaware-densenet121-bhb10k": {
                "module.hidden_representation.weight": (
                    "projection_head.layers.0.weight"
                ),
                "module.features.conv0.weight": (
                    "encoder.features.conv0.weight"
                ),
            },
            "neuro-simclr-resnet18": {
                "module.encoder.conv1.weight": "encoder.conv1.weight",
                "module.projector.2.weight": "projection_head.layers.2.weight",
            },
            "brainiac-vit-b": {
                "backbone.patch_embedding.position_embeddings": (
                    "encoder.pos_embed"
                ),
                "backbone.blocks.3.mlp.linear2.weight": (
                    "encoder.blocks.3.mlp.fc2.weight"
                ),
                "backbone.blocks.3.attn.out_proj.bias": (
                    "encoder.blocks.3.attn.proj.bias"
                ),
                "projection_head.layers.4.running_mean": (
                    "projection_head.layers.4.running_mean"
                ),
            },
        }
        for name, names in expected.items():
            model = self._build(name)
            for original, renamed in names.items():
                with self.subTest(original):
                    for pattern, replacement in self.tool.MODELS[name][
                        "key_mapping"
                    ]:
                        original = re.sub(pattern, replacement, original)
                    self.assertEqual(original, renamed)
                    self.assertIn(renamed, model.state_dict())

    def test_documentation_registry(self):
        """The models advertised in the documentation are the published
        ones."""
        registry = Path(__file__).parents[2] / "doc" / "pretrained_models.json"
        if not registry.is_file():
            self.skipTest("documentation not available")
        models = json.loads(registry.read_text())["models"]
        published = {**self.tool.MODELS, **self.tool.EXTERNAL_MODELS}
        self.assertEqual(
            {m["repo_id"] for m in models},
            {f"nidl-neurospin/{name}" for name in published},
        )
        for model in models:
            with self.subTest(model["repo_id"]):
                info = published[model["repo_id"].split("/")[1]]
                estimator = info["estimator"]
                if not isinstance(estimator, str):
                    estimator = estimator.__name__
                config = info["config"]
                self.assertEqual(model["estimator"], estimator)
                self.assertEqual(
                    model["preprocessing"],
                    config["preprocessing"]["transform"],
                )
                self.assertEqual(
                    model["gated"], "repo_id" in config.get("weights", {})
                )
                for key in ("subjects", "volumes", "modality"):
                    self.assertIn(key, model)
                transform = getattr(
                    preprocessing, model["preprocessing"]
                )
                transform(**_tuples(model["openbhb"]["kwargs"]))

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
        with self.assertRaisesRegex(ValueError, "foreground"):
            CropForeground(margin=4)(np.zeros((1, 4, 4, 4)))
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
