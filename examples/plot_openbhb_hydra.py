"""
How to use NIDL with Hydra
==========================

This tutorial shows how to build a full NIDL experiment using **Hydra
configurations**. Hydra allows you to describe datasets, transforms,
dataloaders, and models directly in YAML and instantiate them at runtime.

The goal is to understand:

- how Hydra instantiates Python objects using ``_target_``.
- how ``${...}`` references work inside configs.
- how YAML anchors (``&name``), aliases (``*name``) and merges (``<<: *name``)
  help avoid duplication.
- how NIDL datasets, transforms, and models can be composed declaratively.


Hydra Concepts Explained
------------------------

**1. ``_target_`` - instantiate Python objects from YAML**

Hydra can instantiate Python objects directly from configuration files.
Any dictionary containing a ``_target_`` key is interpreted as a
description of a Python class or function to be constructed at runtime.

For example::

    encoder:
      _target_: torchvision.ops.MLP
      in_channels: 100
      hidden_channels: [64, 32]

This means: "create an instance of ``torchvision.ops.MLP`` and pass it the
arguments ``in_channels=100`` and ``hidden_channels=[64, 32]``."

During execution, Hydra resolves this block using
``hydra.utils.instantiate``::

    from hydra.utils import instantiate
    encoder = instantiate(cfg.encoder)

This mechanism allows entire components—datasets, transforms,
dataloaders, models—to be declared declaratively in YAML and built
automatically when the experiment runs.

**2. ``${...}`` - reference previously defined config values**

Hydra allows you to reuse values defined elsewhere in the configuration
using the ``${...}`` syntax.

For example::

    argument:
        noise_std: 0.5
    transform:
      _target_: RandomGaussianNoise
      noise_std: ${augment.noise_std}

Here, ``${augment.noise_std}`` is replaced with ``0.5`` during
configuration resolution. The instantiated ``RandomGaussianNoise`` object
therefore receives ``noise_std=0.5`` automatically.

This mechanism ensures that shared parameters (such as augmentation
strengths, dataset paths, or model dimensions) remain synchronized across
the entire Hydra configuration.

**3. YAML anchors ``&name`` and aliases ``*name`` or merges ``<<: *name``**

YAML anchors let you define reusable configuration blocks that can be
referenced later. This keeps Hydra configs concise.

You create an anchor using ``&name``::

    _base_dataset: &base_ds
      _target_: nidl.datasets.TabularDataset
      root: ${data.root}

You can then reuse this block in two different ways.
Using ``*name`` simply inserts the anchored block as a value::

    dataset:
      train: *_base_dataset

Using ``<<: *name`` merges the anchored dictionary into the current one
and allows overriding or adding fields::

    dataset:
      <<: *base_ds
      split: "train"

This means: "start from the contents of ``base_ds`` and then override
the ``split`` field."
"""

# %%
# Imports
# -------
from pathlib import Path
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import numpy as np


# %%
# Transforms
# ----------
# These are simple NumPy‑based transforms used in the Hydra config.

class Flatten:
    def __call__(self, x):
        return x.flatten()


class RandomMask:
    def __init__(self, mask_prob):
        self.mask_prob = mask_prob

    def __call__(self, x):
        mask = (np.random.rand(*x.shape) > self.mask_prob).astype(np.float32)
        return x * mask


class RandomGaussianNoise:
    def __init__(self, noise_std):
        self.noise_std = noise_std

    def __call__(self, x):
        if np.random.rand() > 0.5:
            noise = np.random.randn(*x.shape) * self.noise_std
            return x + noise.astype(np.float32)
        return x


class SBMTransform:
    def __init__(self, channels):
        self.channels = channels

    def __call__(self, x):
        return x[self.channels].flatten()


# %%
# Generate Hydra Config
# ---------------------
# We generate a YAML config so that Sphinx-Gallery can display it.

config_text = r"""
data:
  data_dir: "/tmp/openBHB"
  batch_size: 32
  num_workers: 4

augment:
  mask_prob: 0.8
  noise_std: 0.5
  n_views: 2
  channels:
    - 0
    - 1
    - 2
    - 5

  _base_vbm_transform: &vbm_transform
    _target_: torchvision.transforms.Compose
    transforms:
      - _target_: __main__.Flatten

  _base_sbm_transform: &sbm_transform
    _target_: torchvision.transforms.Compose
    transforms:
      - _target_: __main__.SBMTransform
        channels: ${augment.channels}
      - _target_: __main__.Flatten

  contrast:
    _target_: torchvision.transforms.Compose
    transforms:
      - _target_: __main__.Flatten
      - _target_: __main__.RandomMask
        mask_prob: ${augment.mask_prob}
      - _target_: __main__.RandomGaussianNoise
        noise_std: ${augment.noise_std}

  _base_vbm_ssl_transform: &vbm_ssl_transform
    _target_: nidl.transforms.MultiViewsTransform
    transforms:
      _target_: torchvision.transforms.Compose
      transforms:
        - ${augment._base_vbm_transform}
        - ${augment.contrast}
    n_views: ${augment.n_views}

  _base_sbm_ssl_transform: &sbm_ssl_transform
    _target_: nidl.transforms.MultiViewsTransform
    transforms:
      _target_: torchvision.transforms.Compose
      transforms:
        - ${augment._base_sbm_transform}
        - ${augment.contrast}
    n_views: ${augment.n_views}


dataset:
  _base_openbhb: &openbhb_base
    _target_: nidl.datasets.OpenBHB
    root: ${data.data_dir}
    streaming: false

  ssl_vbm_train:
    <<: *openbhb_base
    target: "age"
    modality: "vbm_roi"
    transforms: *vbm_ssl_transform

  ssl_vbm_val:
    <<: *openbhb_base
    target: "age"
    modality: "vbm_roi"
    split: "val"
    transforms: *vbm_ssl_transform

  ssl_sbm_train:
    <<: *openbhb_base
    target: "age"
    modality: "fs_desikan_roi"
    transforms: *sbm_ssl_transform

  ssl_sbm_val:
    <<: *openbhb_base
    target: "age"
    modality: "fs_desikan_roi"
    split: "val"
    transforms: *sbm_ssl_transform

  vbm_test:
    <<: *openbhb_base
    target: null
    modality: "vbm_roi"
    split: "val"
    transforms: *vbm_transform

  sbm_test:
    <<: *openbhb_base
    target: null
    modality: "fs_desikan_roi"
    split: "val"
    transforms: *sbm_transform

dataloader:
  ssl_vbm_train:
    _target_: torch.utils.data.DataLoader
    dataset: ${dataset.ssl_vbm_train}
    batch_size: ${data.batch_size}
    num_workers: ${data.num_workers}
    shuffle: true

  ssl_vbm_val:
    _target_: torch.utils.data.DataLoader
    dataset: ${dataset.ssl_vbm_val}
    batch_size: ${data.batch_size}
    num_workers: ${data.num_workers}
    shuffle: false

  ssl_sbm_train:
    _target_: torch.utils.data.DataLoader
    dataset: ${dataset.ssl_sbm_train}
    batch_size: ${data.batch_size}
    num_workers: ${data.num_workers}
    shuffle: true

  ssl_sbm_val:
    _target_: torch.utils.data.DataLoader
    dataset: ${dataset.ssl_sbm_val}
    batch_size: ${data.batch_size}
    num_workers: ${data.num_workers}
    shuffle: false

  vbm_test:
    _target_: torch.utils.data.DataLoader
    dataset: ${dataset.vbm_test}
    batch_size: ${data.batch_size}
    num_workers: ${data.num_workers}
    shuffle: false

  sbm_test:
    _target_: torch.utils.data.DataLoader
    dataset: ${dataset.sbm_test}
    batch_size: ${data.batch_size}
    num_workers: ${data.num_workers}
    shuffle: false

model:
  latent_size: 32
  sigma: 4
  max_epochs: 2

  vbm_encoder:
    _target_: torchvision.ops.MLP
    in_channels: 284
    hidden_channels:
      - 64
      - ${model.latent_size}

  sbm_encoder:
    _target_: torchvision.ops.MLP
    in_channels: 272
    hidden_channels:
      - 64
      - ${model.latent_size}

  vbm:
    _target_: nidl.estimators.ssl.YAwareContrastiveLearning
    encoder: ${model.vbm_encoder}
    proj_input_dim: ${model.latent_size}
    proj_hidden_dim: ${model.latent_size}
    proj_output_dim: ${model.latent_size}
    bandwidth: ${model.sigma}
    random_state: 42
    max_epochs: ${model.max_epochs}
    temperature: 0.1
    learning_rate: 1e-5
    enable_checkpointing: false

  sbm:
    _target_: nidl.estimators.ssl.YAwareContrastiveLearning
    encoder: ${model.sbm_encoder}
    proj_input_dim: ${model.latent_size}
    proj_hidden_dim: ${model.latent_size}
    proj_output_dim: ${model.latent_size}
    bandwidth: ${model.sigma}
    random_state: 42
    max_epochs: ${model.max_epochs}
    temperature: 0.1
    learning_rate: 1e-5
    enable_checkpointing: false
"""

tmpdir = Path("/tmp")
config_path = tmpdir / "openbhb_config.yaml"
config_path.write_text(config_text)

print(config_text)


# %%
# Main Experiment
# ---------------
# Hydra loads the configuration file and instantiates all objects.

@hydra.main(
    config_path="/tmp",
    config_name="openbhb_config",
    version_base="1.3",
)
def main(cfg: DictConfig):

    # Instantiate dataloaders
    ssl_vbm_train = instantiate(cfg.dataloader.ssl_vbm_train)
    ssl_vbm_val = instantiate(cfg.dataloader.ssl_vbm_val)
    ssl_sbm_train = instantiate(cfg.dataloader.ssl_sbm_train)
    ssl_sbm_val = instantiate(cfg.dataloader.ssl_sbm_val)
    vbm_test = instantiate(cfg.dataloader.vbm_test)
    sbm_test = instantiate(cfg.dataloader.sbm_test)

    # Instantiate models
    vbm_model = instantiate(cfg.model.vbm)
    sbm_model = instantiate(cfg.model.sbm)

    # Fit models
    vbm_model.fit(ssl_vbm_train, ssl_vbm_val)
    sbm_model.fit(ssl_sbm_train, ssl_sbm_val)

    # Compute embeddings
    z_vbm_test = vbm_model.transform(vbm_test)
    z_sbm_test = sbm_model.transform(sbm_test)

    print(f"Z shapes - VBM: {z_vbm_test.shape}, SBM: {z_sbm_test.shape}")


# %%
# Run Example
# -----------

main()

