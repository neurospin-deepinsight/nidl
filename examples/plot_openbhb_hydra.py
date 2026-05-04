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

config_path = Path(__file__).parent / "openbhb_config.yaml"
print(config_path.read_text())


# %%
# Main experiment
# ---------------
# Hydra loads the configuration file ``openbhb_config.yaml`` located in the
# same directory. All dataloaders, models, and transforms are instantiated
# directly from the config.

@hydra.main(
    config_path=".",
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

