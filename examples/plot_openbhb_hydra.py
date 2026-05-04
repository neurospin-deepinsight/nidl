"""
Presentation of the OpenBHB dataset with Hydra
==============================================

This example revisits the :ref:`sphx_glr_auto_examples_plot_openbhb.py`
example, but this time the full experiment is driven by a
**Hydra configuration**.

Hydra is a configuration framework that lets you:

- compose hierarchical configs (dataset, model, transforms…)
- override any parameter from the command line
- keep experiments reproducible and organized
- instantiate Python objects directly from YAML

In the current example, Hydra is especially useful for:

- managing complex dataloaders and model parameters
- switching between SSL and supervised settings
- enabling large‑scale hyperparameter sweeps
- keeping experiments traceable and structured

Below we show how OpenBHB dataloaders and models can be instantiated
entirely from a Hydra config, and how embeddings are computed.
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
# Custom transforms
# -----------------
# These are simple NumPy‑based transforms used in the OpenBHB config.

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
# Display the Hydra config file
# -----------------------------
# This prints the content of ``openbhb_config.yaml`` so it appears in the
# rendered example.

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
print(config_text)
tmpdir = Path("/tmp")
config_path = tmpdir / "openbhb_config.yaml"
config_path.write_text(config_text)


# %%
# Main experiment
# ---------------
# Hydra loads the configuration file ``openbhb_config.yaml`` located in the
# same directory. All dataloaders, models, and transforms are instantiated
# directly from the config.

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

    print(f"Z shapes — VBM: {z_vbm_test.shape}, SBM: {z_sbm_test.shape}")


# %%
# Run the example
# ---------------

main()

