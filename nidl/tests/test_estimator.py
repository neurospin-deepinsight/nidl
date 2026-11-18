##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import os
import tempfile
from collections import OrderedDict
from functools import partial

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
    DINO,
    DCL,
    IJEPA,
    CoMM,
)
from nidl.backbones.volume.utils import (
    IdentityAdapter,
    Patched3DInputAdapter,
    SimpleFeaturesInputAdapter,
)
from nidl.utils.data_parsing import collate_multimodal
from timm.models.vision_transformer import VisionTransformer
from nidl.estimators.autoencoders import VAE
from nidl.estimators.linear import LogisticRegression
from nidl.losses.yaware_infonce import KernelMetric
from nidl.losses.beta_vae import BetaVAELoss
from nidl.transforms.transforms import MultiViewsTransform
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


class MultimodalTensorDataset(Dataset):
    """ TensorDataset returning a dict {modality: data or None}, where each
    sample misses at most one modality.
    """
    def __init__(self, data, n_modalities=3, transform=None):
        self.data = data
        self.n_modalities = n_modalities
        self.transform = transform

    def __getitem__(self, index):
        sample = {}
        for idx in range(self.n_modalities):
            x = self.data[index]
            if self.transform:
                x = self.transform(x)
            missing = (index % (self.n_modalities + 1) == idx)
            sample[f"mod{idx}"] = None if missing else x
        return sample

    def __len__(self):
        return len(self.data)


class HeterogeneousDataset(Dataset):
    """ Samples {mod0: (25,), mod1: (25,), mod2: (2, 7)}, each sample missing
    at most one modality, with optional views."""
    def __init__(self, n_samples, n_views=None):
        self.n_samples = n_samples
        self.n_views = n_views

    def __getitem__(self, index):
        shapes = {"mod0": (5 * 5,), "mod1": (5 * 5,), "mod2": (2, 7)}
        sample = {}
        for idx, (name, shape) in enumerate(shapes.items()):
            if index % 4 == idx:
                sample[name] = None
            elif self.n_views is None:
                sample[name] = torch.rand(shape)
            else:
                sample[name] = [torch.rand(shape)
                                for _ in range(self.n_views)]
        return sample

    def __len__(self):
        return self.n_samples


class TestEstimators(unittest.TestCase):
    """ Test estimators: simple checks.
    """
    def setUp(self):
        self.latent_size = 10
        self._encoder = nn.Linear(5 * 5, self.latent_size)
        self._encoder_vit = VisionTransformer(
            img_size=5, patch_size=1, in_chans=1, 
            embed_dim=self.latent_size, depth=1, 
            num_heads=2
        )
        self._fc = nn.Linear(self.latent_size, 2)
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
        ssl_2d_dataset = CustomTensorDataset(
            self.fake_data.view(self.n_images, 1, 5, 5),
        )
        multicrop_ssl_dataset = CustomTensorDataset(
            self.fake_data, transform=MultiViewsTransform(ssl_transforms, n_views=6)
        )
        x_dataset = CustomTensorDataset(self.fake_data)
        x_2d_dataset = CustomTensorDataset(self.fake_data.view(self.n_images, 1, 5, 5))
        xy_dataset = CustomTensorDataset(self.fake_data, labels=self.fake_labels)
        xxy_dataset = CustomTensorDataset(
            self.fake_data, labels=self.fake_labels, 
            transform=MultiViewsTransform(ssl_transforms, n_views=2)
        )
        self.ssl_loader = DataLoader(ssl_dataset, batch_size=2, shuffle=False)
        self.ssl_2d_loader = DataLoader(ssl_2d_dataset, batch_size=2, shuffle=False)
        self.multicrop_ssl_loader = DataLoader(multicrop_ssl_dataset, batch_size=2, shuffle=False)
        self.weakly_sup_loader = DataLoader(xxy_dataset, batch_size=2, shuffle=False)
        self.x_loader = DataLoader(x_dataset, batch_size=2, shuffle=False)
        multimodal_ssl_dataset = MultimodalTensorDataset(
            self.fake_data, transform=MultiViewsTransform(ssl_transforms, n_views=2)
        )
        multimodal_x_dataset = MultimodalTensorDataset(self.fake_data)
        collate_fn = partial(
            collate_multimodal, modalities=("mod0", "mod1", "mod2")
        )
        self.multimodal_ssl_loader = DataLoader(
            multimodal_ssl_dataset, batch_size=4, shuffle=False,
            collate_fn=collate_fn
        )
        self.multimodal_x_loader = DataLoader(
            multimodal_x_dataset, batch_size=4, shuffle=False,
            collate_fn=collate_fn
        )
        self.x_2d_loader = DataLoader(x_2d_dataset, batch_size=2, shuffle=False)
        self.xy_loader = DataLoader(xy_dataset, batch_size=2, shuffle=False)

    def ssl_config(self):
        return (
            (
                SimCLR,
                {
                    "encoder": self._encoder,
                    "proj_input_dim": self.latent_size,
                    "proj_hidden_dim": 3,
                    "proj_output_dim": 3,
                    "learning_rate": 5e-4,
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
                    "proj_input_dim": self.latent_size,
                    "proj_hidden_dim": 3,
                    "proj_output_dim": 3,
                    "learning_rate": 5e-4,
                    "lambd": 0.005,
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
                    "proj_input_dim": self.latent_size,
                    "proj_hidden_dim": 3,
                    "proj_output_dim": 3,
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
                    "proj_input_dim": self.latent_size,
                    "proj_hidden_dim": 3,
                    "proj_output_dim": 3,
                    "lr_scheduler": "none",
                    "temperature": 0.1,
                    "learning_rate": 5e-4,
                    "weight_decay": 1e-4,
                    "max_epochs": 2,
                    "limit_train_batches": 3
                },
            ),
            (
                DCL,
                {
                    "encoder": self._encoder,
                    "proj_input_dim": self.latent_size,
                    "proj_hidden_dim": 3,
                    "proj_output_dim": 3,
                    "learning_rate": 5e-4,
                    "temperature": 0.07,
                    "weight_decay": 1e-4,
                    "max_epochs": 2,
                    "random_state": 42,
                    "limit_train_batches": 3,
                },
            ),
            (
                IJEPA,
                {
                    "encoder": self._encoder_vit,
                    "dim": 2,
                    "min_keep": 0,
                    "learning_rate": 5e-4,
                    "weight_decay": 1e-4,
                    "max_epochs": 2,
                    "random_state": 42,
                    "limit_train_batches": 3,
                },
            )
        )

    def multicrop_ssl_config(self):
        return ((
            DINO,
                {
                    "encoder": self._encoder,
                    "proj_input_dim": self.latent_size,
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
                    "proj_input_dim": self.latent_size,
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

    def multimodal_ssl_config(self):
        params = {
            "num_modalities": 3,
            "embed_dim": self.latent_size,
            "num_heads": 2,
            "proj_hidden_dim": 3,
            "proj_output_dim": 3,
            "learning_rate": 5e-4,
            "max_epochs": 2,
            "random_state": 42,
            "limit_train_batches": 3,
        }

        def map_encoder():
            # Backbone returning 3D feature maps of shape (1, 5, 5, 1)
            return nn.Sequential(
                nn.Linear(5 * 5, 5 * 5), nn.Unflatten(1, (1, 5, 5, 1))
            )

        return (
            (
                CoMM,
                {
                    **params,
                    "encoder": nn.Linear(5 * 5, self.latent_size),
                    "freeze_encoder": False,
                }
            ),
            (
                CoMM,
                {
                    **params,
                    "encoder": nn.Linear(5 * 5, self.latent_size),
                    "freeze_encoder": True,
                }
            ),
            (
                CoMM,
                {
                    **params,
                    "encoder": [map_encoder(), map_encoder()],
                    "mod_to_encoder": [0, 1, 1],
                    "adapter": Patched3DInputAdapter(
                        num_channels=1, stride_level=1, patch_size_full=1,
                        dim_tokens=self.latent_size, volume_size=(5, 5, 1)
                    ),
                    "per_modality_adapter": True,
                    "freeze_encoder": False,
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
                    "proj_input_dim": self.latent_size,
                    "proj_hidden_dim": 10,
                    "proj_output_dim": 10,
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
                    "proj_input_dim": self.latent_size,
                    "proj_hidden_dim": 10,
                    "proj_output_dim": 10,
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
            if isinstance(model, IJEPA): # IJEPA is designed for 2D/3D inputs only
                ssl_loader = self.ssl_2d_loader
                x_loader = self.x_2d_loader
            else:
                ssl_loader = self.ssl_loader
                x_loader = self.x_loader
            model.fit(ssl_loader)
            z = model.transform(x_loader)
            self.assertTrue(
                z.shape == (self.n_images, self.latent_size),
                msg="Shape mismatch for transformed data: "
                    f"{z.shape} != {(self.n_images, self.latent_size)}",
            )

    def test_multicrop_ssl(self):
        """ Test self supervised model with multi-crop (simple check)."""
        for klass, params in self.multicrop_ssl_config():
            print(f"[{print_multicolor(klass.__name__, display=False)}]...")
            model = klass(**params)
            model.fit(self.multicrop_ssl_loader)
            z = model.transform(self.x_loader)
            self.assertTrue(
                z.shape == (self.n_images, self.latent_size),
                msg="Shape mismatch for transformed data: "
                    f"{z.shape} != {(self.n_images, self.latent_size)}",
            )

    def test_multimodal_ssl(self):
        """ Test multimodal self supervised model (simple check)."""
        for klass, params in self.multimodal_ssl_config():
            print(f"[{print_multicolor(klass.__name__, display=False)}]...")
            model = klass(**params)
            model.fit(self.multimodal_ssl_loader)
            z = model.transform(self.multimodal_x_loader)
            self.assertTrue(
                z.shape == (self.n_images, self.latent_size),
                msg="Shape mismatch for transformed data: "
                    f"{z.shape} != {(self.n_images, self.latent_size)}",
            )
        # The loss terms are logged with the names of the modalities
        _, params = self.multimodal_ssl_config()[0]
        model = CoMM(**params, modality_names=("t1", "t2", "flair"))
        model.fit(self.multimodal_ssl_loader)
        for name in ("t1", "t2", "flair", "fused"):
            for metric in ("ssl_loss", "ssl_acc"):
                self.assertIn(f"{metric}_{name}/train",
                              model.trainer.callback_metrics)

    def test_multimodal_ssl_bad_parameters(self):
        """ Test multimodal self supervised model with bad parameters."""
        _, params = self.multimodal_ssl_config()[0]
        # Several encoders without, or with a wrong, `mod_to_encoder`
        for mod_to_encoder in (None, [0, 1], [0, 1, 2]):
            with self.assertRaises(ValueError):
                CoMM(**{
                    **params,
                    "encoder": [nn.Linear(5 * 5, self.latent_size),
                                nn.Linear(5 * 5, self.latent_size)],
                    "mod_to_encoder": mod_to_encoder,
                })
        # `sync_batchnorm` with several trainable encoders having batch norm
        encoders = [
            nn.Sequential(nn.Linear(5 * 5, self.latent_size),
                          nn.BatchNorm1d(self.latent_size))
            for _ in range(2)
        ]
        with self.assertRaises(ValueError):
            CoMM(**{
                **params,
                "encoder": encoders,
                "mod_to_encoder": [0, 1, 1],
                "sync_batchnorm": True,
            })
        # Scans of different shapes within a modality
        for other in (torch.zeros(1, 6, 6, 6), torch.zeros(3, 8, 8, 8)):
            with self.assertRaises(ValueError):
                collate_multimodal(
                    [{"t1": torch.zeros(1, 8, 8, 8)}, {"t1": other}],
                    modalities=("t1",),
                )
        # A list of adapters of the wrong length, or with
        # `per_modality_adapter`
        adapter = SimpleFeaturesInputAdapter()
        for extra in ({"adapter": [adapter, adapter]},
                      {"adapter": [adapter] * 3,
                       "per_modality_adapter": True}):
            with self.assertRaises(ValueError):
                CoMM(**{**params, **extra})
        # Wrong number of, duplicated or reserved modality names
        for names in (("t1", "t2"), ("t1", "t1", "t2"),
                      ("t1", "t2", "fused")):
            with self.assertRaises(ValueError):
                CoMM(**{**params, "modality_names": names})
        # `embed_dim` not divisible by `num_heads`
        with self.assertRaises(ValueError):
            CoMM(**{**params, "embed_dim": 7})
        # `embed_dim` different from the token dimension of the adapter
        batch = next(iter(self.multimodal_x_loader))
        model = CoMM(**{**params, "embed_dim": 12})
        with self.assertRaises(ValueError):
            model.transform_step(batch, 0)
        # More modality slots in the batches than in the estimator
        model = CoMM(**{**params, "num_modalities": 2})
        with self.assertRaises(ValueError):
            model.transform_step(batch, 0)
        # `transform` expects a single image per modality, not views
        model = CoMM(**params)
        model.fit(self.multimodal_ssl_loader)
        with self.assertRaises(ValueError):
            model.transform(self.multimodal_ssl_loader)

    def _multimodal_batch(self, present, scans=None):
        """ Two views of a batch, given the (B, M) bool presence mask and
        optionally the (2, B, M, 25) scans of every (sample, modality)."""
        if scans is None:
            scans = torch.rand(2, *present.shape, 5 * 5)
        sample_idx = [
            present[:, m].nonzero(as_tuple=True)[0] if present[:, m].any()
            else None
            for m in range(present.shape[1])
        ]
        views = [
            [None if idx is None else view[idx, m]
             for m, idx in enumerate(sample_idx)]
            for view in scans
        ]
        return {"views": views, "sample_idx": sample_idx,
                "n_samples": len(present)}

    def test_multimodal_ssl_unused_modules(self):
        """ Test that the adapters and encoders of a modality absent from a
        batch still get a (zero) gradient, as required by DDP."""
        _, params = self.multimodal_ssl_config()[0]
        present = torch.ones(8, 3, dtype=torch.bool)
        present[:, 2] = False             # no scan of modality 2
        for extra in (
            {"encoder": [nn.Linear(5 * 5, self.latent_size),
                         nn.Linear(5 * 5, self.latent_size)],
             "mod_to_encoder": [0, 0, 1]},
            {"adapter": nn.Sequential(
                nn.Linear(self.latent_size, self.latent_size),
                nn.Unflatten(1, (1, self.latent_size))),
             "per_modality_adapter": True},
        ):
            model = CoMM(**{**params, **extra})
            z1, z2, present_ = model(self._multimodal_batch(present))
            model.loss(z1, z2, present_)["loss"].backward()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIsNotNone(
                        param.grad, msg=f"{name} did not get a gradient")

    def test_multimodal_ssl_missing_rows_in_head(self):
        """ Test that the per-modality projections of the samples having a
        modality do not depend on the samples missing it."""
        _, params = self.multimodal_ssl_config()[0]
        model = CoMM(**params).train()
        scans = torch.rand(2, 8, 3, 5 * 5)
        # samples 4-7 miss modality 1
        present = torch.ones(8, 3, dtype=torch.bool)
        present[4:, 1] = False
        z_missing, _, present_ = model(self._multimodal_batch(present, scans))
        # samples 0-3 alone
        z_alone, _, _ = model(self._multimodal_batch(present[:4],
                                                     scans[:, :4]))
        self.assertTrue(torch.equal(present_, present))
        # modality 1 of samples 0-3: the absent rows of samples 4-7 do not
        # enter the batch normalization of the head
        self.assertTrue(torch.allclose(z_missing[1][:4], z_alone[1],
                                       atol=1e-5))
        self.assertTrue(torch.equal(z_missing[1][4:],
                                    torch.zeros_like(z_missing[1][4:])))
        # a modality held by a single sample is dropped during training
        present[1:, 2] = False
        _, _, present_ = model(self._multimodal_batch(present, scans))
        self.assertFalse(present_[:, 2].any())

    def test_multimodal_ssl_heterogeneous(self):
        """ Test modalities of different shapes, encoded by different
        backbones into different numbers of tokens."""
        _, params = self.multimodal_ssl_config()[0]
        # mod0 / mod1: vectors (25,) -> 1 token; mod2: (2, 7) -> 2 tokens
        collate = partial(collate_multimodal,
                          modalities=("mod0", "mod1", "mod2"))
        ssl_loader = DataLoader(
            HeterogeneousDataset(self.n_images, n_views=2), batch_size=4,
            collate_fn=collate)
        x_loader = DataLoader(
            HeterogeneousDataset(self.n_images), batch_size=4,
            collate_fn=collate)
        model = CoMM(**{
            **params,
            "encoder": [nn.Linear(5 * 5, self.latent_size),
                        nn.Linear(7, self.latent_size)],
            "mod_to_encoder": [0, 0, 1],
            "adapter": [SimpleFeaturesInputAdapter(),
                        SimpleFeaturesInputAdapter(), IdentityAdapter()],
        })
        model.fit(ssl_loader)
        z = model.transform(x_loader)
        self.assertEqual(tuple(z.shape), (self.n_images, self.latent_size))
        batch = next(iter(x_loader))
        tokens = model.encoder.encode(batch["x"])
        self.assertEqual([tok.shape[1] for tok in tokens], [1, 1, 2])

    def test_multimodal_ssl_sync_batchnorm_calls(self):
        """ Test that a backbone with synchronized batch normalization must
        run exactly once per batch."""
        _, params = self.multimodal_ssl_config()[0]
        model = CoMM(**{
            **params,
            "encoder": nn.Sequential(nn.Linear(5 * 5, self.latent_size),
                                     nn.BatchNorm1d(self.latent_size)),
        })
        model.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(
            model.encoder).train()
        model.encoder._check_sync_batchnorm_calls([0])
        for calls in ([], [0, 0]):
            with self.assertRaises(RuntimeError):
                model.encoder._check_sync_batchnorm_calls(calls)

    def test_multimodal_ssl_checkpoint(self):
        """ Test that a CoMM checkpoint reloads when the encoder is given as
        a class."""
        _, params = self.multimodal_ssl_config()[0]
        params = {
            **params,
            "encoder": nn.Linear,
            "encoder_kwargs": {"in_features": 5 * 5,
                               "out_features": self.latent_size},
            "enable_checkpointing": False,
            "logger": False,
        }
        model = CoMM(**params)
        model.fit(self.multimodal_ssl_loader)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "comm.ckpt")
            model.trainer.save_checkpoint(path)
            # the hyper-parameters hold the encoder class
            reloaded = CoMM.load_from_checkpoint(path, weights_only=False)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, reloaded.state_dict()[key]))

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
                z.shape == (self.n_images, self.latent_size),
                msg="Shape mismatch for transformed data: "
                    f"{z.shape} != {(self.n_images, self.latent_size)}",
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
