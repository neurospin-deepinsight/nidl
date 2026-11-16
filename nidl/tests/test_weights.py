##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from nidl.utils.weights import Weights


class Net(nn.Module):
    """Small model with an encoder and a decoder submodule."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 3)
        self.decoder = nn.Linear(3, 4)


def _assert_state_equal(test, module, state_dict, prefix=""):
    for key, value in module.state_dict().items():
        test.assertTrue(torch.equal(value, state_dict[prefix + key]), key)


class TestWeights(unittest.TestCase):
    """Test the `Weights` class."""

    def setUp(self):
        # No test may reach the network (Hugging Face or NeuroSpin hub).
        for name, target in (
            ("hf_hub_download", "huggingface_hub.hf_hub_download"),
            ("urlretrieve", "urllib.request.urlretrieve"),
        ):
            patcher = patch(target, side_effect=ConnectionError("offline"))
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        torch.manual_seed(0)
        self.source = Net()
        self.state = {
            k: v.clone() for k, v in self.source.state_dict().items()
        }
        torch.save(self.state, self.dir / "weights.pt")
        torch.save(
            {"state_dict": self.state, "epoch": 3, "hyper_parameters": {}},
            self.dir / "model.ckpt",
        )

    def _weights(self, filepath="weights.pt", **kwargs):
        return Weights(str(self.dir), filepath, **kwargs)

    # Construction

    def test_local_without_data_dir(self):
        weights = self._weights()
        self.assertEqual(weights.dtype, "local")
        self.assertIsNone(weights.data_dir)
        self.assertEqual(weights.weight_file, self.dir / "weights.pt")
        self.assertFalse(weights.is_lightning_ckpt)

    def test_local_with_data_dir(self):
        weights = self._weights(data_dir=self.dir)
        self.assertEqual(weights.data_dir, self.dir)

    def test_ckpt_flag(self):
        self.assertTrue(self._weights("model.ckpt").is_lightning_ckpt)

    def test_missing_file(self):
        with self.assertRaises(AssertionError):
            self._weights("missing.pt")

    def test_arguments_order(self):
        weights = Weights(str(self.dir), "weights.pt", str(self.dir))
        self.assertEqual(weights.filepath, "weights.pt")
        self.assertEqual(weights.data_dir, self.dir)

    def test_hub_requires_data_dir(self):
        for hub in ("hf-hub:org/model", "ns-hub:org/model"):
            with self.assertRaises(AssertionError) as cm:
                Weights(hub, "weights.pt")
            msg = str(cm.exception)
            self.assertIn("data_dir", msg)
            self.assertIn("downloading weights", msg)

    # Downloads (mocked at the library boundary, like in test_datasets.py)

    def test_hf_hub_download(self):
        self.hf_hub_download.side_effect = None
        self.hf_hub_download.return_value = str(self.dir / "weights.pt")
        weights = Weights("hf-hub:org/model@v1", "weights.pt", self.dir)
        self.hf_hub_download.assert_called_once_with(
            repo_id="org/model",
            filename="weights.pt",
            revision="v1",
            repo_type="model",
            local_dir=str(self.dir),
            force_download=False,
        )
        self.assertEqual(weights.dtype, "hf-hub")
        self.assertEqual(weights.weight_file, self.dir / "weights.pt")
        self.urlretrieve.assert_not_called()

    def test_hf_hub_download_without_revision(self):
        self.hf_hub_download.side_effect = None
        self.hf_hub_download.return_value = str(self.dir / "weights.pt")
        Weights("hf-hub:org/model", "weights.pt", self.dir)
        self.assertIsNone(self.hf_hub_download.call_args.kwargs["revision"])

    def test_hf_download_force(self):
        self.hf_hub_download.side_effect = None
        self.hf_hub_download.return_value = str(self.dir / "weights.pt")
        out = Weights.hf_download(
            self.dir, "org/model", "weights.pt", force_download=True
        )
        self.assertEqual(out, self.dir / "weights.pt")
        self.assertTrue(self.hf_hub_download.call_args.kwargs["force_download"])

    def test_hf_download_import_error(self):
        # Simulate huggingface_hub not being installed
        with (
            patch.dict(sys.modules, {"huggingface_hub": None}),
            self.assertRaisesRegex(
                ModuleNotFoundError, "pip install huggingface_hub"
            ),
        ):
            Weights.hf_download(self.dir, "org/model", "weights.pt")

    def test_hf_weights_can_be_loaded(self):
        self.hf_hub_download.side_effect = None
        self.hf_hub_download.return_value = str(self.dir / "weights.pt")
        weights = Weights("hf-hub:org/model", "weights.pt", self.dir)
        model = Net()
        weights.load_pretrained(model)
        _assert_state_equal(self, model, self.state)

    def _fake_urlretrieve(self, url, filename):
        Path(filename).write_bytes(b"downloaded")

    def test_ns_hub_download(self):
        cache = self.dir / "cache"
        self.urlretrieve.side_effect = self._fake_urlretrieve
        weights = Weights("ns-hub:org/model", "sub/weights.pt", cache)
        expected = cache / "org" / "model" / "sub" / "weights.pt"
        self.assertEqual(weights.dtype, "ns-hub")
        self.assertEqual(weights.weight_file, expected)
        self.assertTrue(expected.is_file())
        self.urlretrieve.assert_called_once()
        url, filename = self.urlretrieve.call_args.args
        self.assertTrue(url.startswith(Weights.NS_URL))
        self.assertEqual(filename, str(expected))
        self.hf_hub_download.assert_not_called()

    def test_ns_hub_download_skipped_if_cached(self):
        cached = self.dir / "org" / "model" / "weights.pt"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"cached")
        weights = Weights("ns-hub:org/model", "weights.pt", self.dir)
        self.assertEqual(weights.weight_file, cached)
        self.urlretrieve.assert_not_called()

    def test_ns_download_force(self):
        cached = self.dir / "org" / "model" / "weights.pt"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"cached")
        self.urlretrieve.side_effect = self._fake_urlretrieve
        Weights.ns_download(self.dir, "org/model", "weights.pt", True)
        self.urlretrieve.assert_called_once()
        self.assertEqual(cached.read_bytes(), b"downloaded")

    def test_hub_split(self):
        self.assertEqual(
            Weights.hub_split("hf-hub:org/model@v1"), ("org/model", "v1")
        )
        self.assertEqual(
            Weights.hub_split("ns-hub:org/model"), ("org/model", None)
        )

    # load_pretrained

    def test_load_pretrained_full(self):
        model = Net()
        self._weights().load_pretrained(model)
        _assert_state_equal(self, model, self.state)

    def test_load_pretrained_from_ckpt_warns(self):
        model = Net()
        with self.assertWarnsRegex(UserWarning, "load_checkpoint"):
            self._weights("model.ckpt").load_pretrained(model)
        _assert_state_equal(self, model, self.state)

    def test_warning_message_is_readable(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._weights("model.ckpt").load_pretrained(Net())
        self.assertIn("Lightning or nidl checkpoint", str(caught[0].message))

    def test_no_warning_for_plain_weights(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._weights().load_pretrained(Net())

    def test_load_submodule(self):
        encoder = nn.Linear(4, 3)
        self._weights().load_pretrained(encoder, submodule_prefix="encoder.")
        _assert_state_equal(self, encoder, self.state, prefix="encoder.")

    def test_load_submodule_from_ckpt_does_not_warn(self):
        decoder = nn.Linear(3, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._weights("model.ckpt").load_pretrained(
                decoder, submodule_prefix="decoder"
            )
        _assert_state_equal(self, decoder, self.state, prefix="decoder.")

    def test_prefix_without_trailing_dot(self):
        encoder = nn.Linear(4, 3)
        self._weights().load_pretrained(encoder, submodule_prefix="encoder")
        _assert_state_equal(self, encoder, self.state, prefix="encoder.")

    def test_prefix_does_not_match_partial_name(self):
        # "enc" must not select "encoder.*" keys.
        with self.assertRaises(ValueError):
            self._weights().load_pretrained(
                nn.Linear(4, 3), submodule_prefix="enc"
            )

    def test_prefix_only_stripped_at_start(self):
        # The prefix also appears inside the key ("encoder.encoder.weight").
        state = {
            "encoder.encoder.weight": torch.randn(2, 2),
            "encoder.encoder.bias": torch.randn(2),
            "decoder.weight": torch.randn(2, 2),
        }
        torch.save(state, self.dir / "nested.pt")
        outer = nn.Module()
        outer.encoder = nn.Linear(2, 2)
        self._weights("nested.pt").load_pretrained(
            outer, submodule_prefix="encoder"
        )
        self.assertTrue(
            torch.equal(outer.encoder.weight, state["encoder.encoder.weight"])
        )
        self.assertTrue(
            torch.equal(outer.encoder.bias, state["encoder.encoder.bias"])
        )

    def test_unknown_prefix(self):
        with self.assertRaises(ValueError) as cm:
            self._weights().load_pretrained(
                nn.Linear(4, 3), submodule_prefix="projector"
            )
        msg = str(cm.exception)
        self.assertIn("projector.", msg)
        self.assertIn("'decoder', 'encoder'", msg)
        self.assertIn("Prefixes in state_dict", msg)

    def test_mismatching_keys_raise(self):
        with self.assertRaises(RuntimeError):
            self._weights().load_pretrained(nn.Linear(4, 3))

    def test_map_location(self):
        model = Net()
        with patch(
            "nidl.utils.weights.torch.load", return_value=self.state
        ) as load:
            self._weights().load_pretrained(
                model, map_location="cpu", weights_only=False
            )
        load.assert_called_once_with(
            self.dir / "weights.pt", weights_only=False, map_location="cpu"
        )

    def test_default_load_arguments(self):
        with patch(
            "nidl.utils.weights.torch.load", return_value=self.state
        ) as load:
            self._weights().load_pretrained(Net())
        load.assert_called_once_with(
            self.dir / "weights.pt", weights_only=True, map_location="cpu"
        )

    def test_missing_weight_file_warns(self):
        weights = self._weights()
        weights.weight_file = None
        with self.assertWarnsRegex(UserWarning, "weight file"):
            self.assertIsNone(weights.load_pretrained(Net()))

    # load_checkpoint

    def test_load_checkpoint_not_a_ckpt_warns(self):
        with self.assertWarnsRegex(UserWarning, "Lightning's checkpoint"):
            self.assertIsNone(self._weights().load_checkpoint(MagicMock()))

    def test_load_checkpoint_default_devices(self):
        model_cls = MagicMock()
        weights = self._weights("model.ckpt")
        out = weights.load_checkpoint(model_cls, lr=0.1)
        model_cls.load_from_checkpoint.assert_called_once_with(
            checkpoint_path=weights.weight_file,
            map_location="cpu",
            lr=0.1,
            devices="auto",
            accelerator="auto",
        )
        self.assertIs(out, model_cls.load_from_checkpoint.return_value)

    def test_load_checkpoint_keeps_user_devices(self):
        model_cls = MagicMock()
        self._weights("model.ckpt").load_checkpoint(
            model_cls, devices=1, accelerator="cpu"
        )
        kwargs = model_cls.load_from_checkpoint.call_args.kwargs
        self.assertEqual(kwargs["devices"], 1)
        self.assertEqual(kwargs["accelerator"], "cpu")

    def test_repr(self):
        self.assertIn("weights.pt", repr(self._weights(data_dir=self.dir)))


if __name__ == "__main__":
    unittest.main()