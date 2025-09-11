"""
Multi Channels VAE (MCVAE)
==========================

Source: https://github.com/ggbioing/mcvae/tree/master/examples/mcvae

The Multi Channel VAE (MCVAE) is an extension of the variational autoencoder
able to jointly model multiple data source named channels.


Import
------
"""

import os
import sys
import time
import copy
import itertools

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from nidl.utils import Bunch, Weights
from nidl.estimators.ae import MCVAE
from nidl.callbacks.caching import CachingCallback


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


# %%
# Utils
# -----
#
# Define some tools.


class GeneratorUniform(nn.Module):
    """ Generate multiple sources (channels) of data through a linear
    generative model:
    z ~ N(0,I)
    for c_idx in n_channels:
        x_ch = W_ch(c_idx)
    where 'W_ch' is an arbitrary linear mapping z -> x_ch
    """
    def __init__(self, lat_dim=2, n_channels=2, n_feats=5, seed=100):
        super(GeneratorUniform, self).__init__()
        self.lat_dim = lat_dim
        self.n_channels = n_channels
        self.n_feats = n_feats
        self.seed = seed
        np.random.seed(self.seed)

        W = []
        for c_idx in range(n_channels):
            w_ = np.random.uniform(-1, 1, (self.n_feats, lat_dim))
            u, s, vt = np.linalg.svd(w_, full_matrices=False)
            w = (u if self.n_feats >= lat_dim else vt)
            W.append(torch.nn.Linear(lat_dim, self.n_feats, bias=False))
            W[c_idx].weight.data = torch.FloatTensor(w)

        self.W = torch.nn.ModuleList(W)

    def forward(self, z):
        if isinstance(z, list):
            return [self.forward(_) for _ in z]
        if type(z) == np.ndarray:
            z = torch.FloatTensor(z)
        assert z.size(1) == self.lat_dim
        obs = []
        for ch in range(self.n_channels):
            x = self.W[ch](z)
            obs.append(x.detach())
        return obs


class SyntheticDataset(Dataset):
    def __init__(self, n_samples=500, lat_dim=2, n_feats=5, n_channels=2,
                 generatorclass=GeneratorUniform, snr=1, train=True):
        super(SyntheticDataset, self).__init__()
        self.n_samples = n_samples
        self.lat_dim = lat_dim
        self.n_feats = n_feats
        self.n_channels = n_channels
        self.snr = snr
        self.train = train
        seed = (7 if self.train else 14)
        np.random.seed(seed)
        self.z_ = np.random.normal(size=(self.n_samples, self.lat_dim))
        self.generator = generatorclass(
            lat_dim=self.lat_dim, n_channels=self.n_channels,
            n_feats=self.n_feats)
        self.x_ = self.generator(self.z_)
        self.X, self.X_noisy = preprocess_and_add_noise(self.x_, snr=snr)
        self.X = [np.expand_dims(x.astype(np.float32), axis=1)
                  for x in self.X]
        self.X_noisy = [np.expand_dims(x.astype(np.float32), axis=1)
                  for x in self.X_noisy]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return [x[item] for x in self.X]

    @property
    def shape(self):
        return (len(self), len(self.X))


def preprocess_and_add_noise(x, snr, seed=0):
    if not isinstance(snr, list):
        snr = [snr] * len(x)
    scalers = [StandardScaler().fit(c_arr) for c_arr in x]
    x_std = [scalers[c_idx].transform(x[c_idx]) for c_idx in range(len(x))]
    # seed for reproducibility in training/testing based on prime number basis
    seed = (seed + 3 * int(snr[0] + 1) + 5 * len(x) + 7 * x[0].shape[0] +
            11 * x[0].shape[1])
    np.random.seed(seed)
    x_std_noisy = []
    for c_idx, arr in enumerate(x_std):
        sigma_noise = np.sqrt(1. / snr[c_idx])
        x_std_noisy.append(arr + sigma_noise * np.random.randn(*arr.shape))
    return x_std, x_std_noisy


class DropoutRateCallback(pl.Callback):
    """ Display the dropout rate at each epoch.

    Parameters
    ----------
    fig: Figure
        the current figure.
    sleep: float, default=0.001
        the sleep time between each refresh.
    """
    def __init__(self, sleep=0.001):
        super().__init__()
        self.fig, self.ax = None, None
        self.sleep = sleep

    def on_train_batch_end(self, trainer, pl_module, outputs, batch,
                           batch_idx):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots()
        if pl_module.current_epoch % 10 == 0:
            do = np.sort(pl_module.dropout.numpy().reshape(-1))
            self.ax.clear()
            self.ax.bar(range(len(do)), do)
            self.ax.spines[["right", "top"]].set_visible(False)
            self.fig.suptitle(
                f"Dropout probability of {model_params.latent_dim} fitted latent "
                "dimensions in sMCVAE",
                fontweight="bold"
            )
            self.ax.set_title(f"{data_params.lat_dim} true latent dimensions")
            self.fig.canvas.draw()
            plt.draw()
            plt.pause(self.sleep)


def plot_data(X, ch_idx=-1, title=None):
    """ Display multidimentional data using a pairplot.
    """
    assert isinstance(X, list)
    data = X[ch_idx]
    df = pd.DataFrame(
        data.squeeze(),
        columns=[f"feat{idx + 1}" for idx in range(data.shape[-1])]
    )
    g = sns.pairplot(
        df,
        diag_kind=None,
        corner=True
    )
    if title is not None:
        g.fig.suptitle(title, fontweight="bold")


# %%
# Create synthetic data
# ---------------------
#
# Generate multiple sources (channels) of data through a linear generative
# model.

data_params = Bunch(
    n_samples=500,
    lat_dim=2,
    n_feats=4,
    n_channels=3,
    snr=10,
)
datasets = {
    split: SyntheticDataset(
        train=(split == "train"),
        **data_params,
    )
    for split in ["train", "validation"]
}
print(f"-- input train: {[ch.shape for ch in datasets['train'].X]}")
print(f"-- input validation: {[ch.shape for ch in datasets['validation'].X]}")
dataloaders = {
    split: torch.utils.data.DataLoader(
        datasets[split],
        batch_size=data_params.n_samples,
        shuffle=(split == "train"),
        num_workers=1,
    )
    for split in ["train", "validation"]
}


# %%
# Sparse vs non sparse
# --------------------
#
# Train a sparse and a non sparse MCVAE.
#
# As in many tutorials before, we provide pre-trained models.

load_pretrained = True
checkpointdir = "/tmp/checkpoints"
weights = {
    name: Weights(
        name="hf-hub:neurospin/mcvae-synth",
        data_dir=checkpointdir,
        filepath=f"{name}-synth.ckpt"
    ) for name in ("sMCVAE", "MCVAE")
}
if load_pretrained:
    for _, weight in weights.items():
        src = weight.weight_file
        dst = os.path.join(checkpointdir, os.path.basename(src))
        if not os.path.exists(dst):
            os.symlink(src, dst)

model_params = Bunch(
    latent_dim=5,
    n_feats=[data_params.n_feats] * data_params.n_channels,
    vae_kwargs={},
    lr=2e-3,
    weight_decay=0,
    max_epochs=10000,
    log_every_n_steps=1,
    random_state=42,
    logger=False,
)
callbacks = {
    name: [
        ModelCheckpoint(
            dirpath=checkpointdir,
            filename=f"{name}-synth",
            save_weights_only=False,
            save_last=False,
            save_top_k=1,
            every_n_epochs=model_params.max_epochs,
            enable_version_counter=False
        ),
        CachingCallback(),
        DropoutRateCallback() if name == "sMCVAE" else None
    ] for name in ("sMCVAE", "MCVAE")
}
models = {
    name: MCVAE(
        sparse=(name == "sMCVAE"),
        dropout_threshold=(None if name == "MCVAE" else 0.2),
        callbacks=(callbacks[name][:2] if name == "MCVAE" else callbacks[name]),
        **model_params,
    )
    for name in ("sMCVAE", "MCVAE")
}
for name, model in models.items():
    save_callback, cache_callback = model.trainer_params["callbacks"][:2]
    save_path = save_callback.format_checkpoint_name(metrics={})
    if not os.path.isfile(save_path):
        model.fit(dataloaders["train"], dataloaders["validation"])
        print(f"Saving {name}: {save_callback.last_model_path}...")
        _weights = {
            "model": model.state_dict(),
            "cache": cache_callback.state_dict()
        }
        torch.save(_weights, save_path)
    else:
        print(f"Restoring {name}: {save_path}...")
        _weights = torch.load(save_path, weights_only=False)
        model.load_state_dict(_weights["model"])
        cache_callback.load_state_dict(_weights["cache"])
        model.cache = cache_callback.returned_objects
        model.fitted_ = True
        models[name] = model

        
# %%
# Display results
# ---------------
#
# First store latent varaibles, generative parameters, and reconstructed data.

plt.close("all")
z = {}     # Latent Space
g = {}     # Generative Parameters
x_hat = {}  # Reconstructed channels
for name, model in models.items():
    print(name)
    print(model.cache)
    q = model.cache.validation[0]
    print(f"-- encoded distribution q(z|x): {q}")
    z[name] = model.p_to_prediction(q)
    print(f"-- z: {[_z.shape for _z in z[name]]}")
    if model.sparse:
        z[name] = model.apply_threshold(z[name])
    z[name] = np.array(z[name]).reshape(-1)
    print(f"-- z flat: {z[name].shape}")
    g[name] = [
        model.vae[c_idx].encode.w_mu.weight.detach().cpu().numpy()
        for c_idx in range(model.n_channels)
    ]
    g[name] = np.array(g[name]).reshape(-1)
    print(f"-- g flat: {g[name].shape}")
    model.return_reconstructions = True
    p = model.transform(dataloaders["validation"])[0]
    x_hat[name] = model.reconstruct(p)


# %%
# Compare the generated latent spaces and reconstructions. With such a simple 
# dataset, MCVAE and sparse-MCVAE gives the same results in terms of latent
# space and generative parameters.

plot_data(datasets["validation"].X, title=f"Ground truth")
plot_data(
    datasets["validation"].X_noisy,
    title=f"ENoisy data fitted by the models (snr={data_params.snr})"
)
for name in models:
    plot_data(x_hat[name], title=f"Reconstructed with {name} model")


plt.figure()
ax = plt.subplot(1,2,1)
ax.spines[["right", "top"]].set_visible(False)
plt.hist([z["sMCVAE"], z["MCVAE"]], bins=20, color=["k", "gray"])
plt.legend(["Sparse", "Non sparse"], loc="upper right")
plt.title("Latent dimensions\n distribution")
plt.ylabel("Count")
plt.xlabel("Value")
ax = plt.subplot(1,2,2)
ax.spines[["right", "top"]].set_visible(False)
plt.hist([g["sMCVAE"], g["MCVAE"]], bins=20, color=["k", "gray"])
plt.title(
    "Generative parameters\n "
    r"$\mathbf{\theta} = \{\mathbf{\theta}_1 \ldots \mathbf{\theta}_C\}$"
)
plt.xlabel("Value")


# %%
# However, only with the sparse model is able to easily identify the
# important latent dimensions.

do = np.sort(models["sMCVAE"].dropout.numpy().reshape(-1))
plt.figure()
plt.bar(range(len(do)), do)
plt.axhline(
    y=models["sMCVAE"].hparams.dropout_threshold, color="r", linestyle="--")
plt.gca().spines[["right", "top"]].set_visible(False)
plt.suptitle(
    f"Dropout probability of {model_params.latent_dim} fitted latent "
    "dimensions in sMCVAE",
    fontweight="bold"
)
plt.title(f"{data_params.lat_dim} true latent dimensions")

plt.show()
