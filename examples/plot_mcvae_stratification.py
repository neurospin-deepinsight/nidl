"""
Multi-modal Latent Variable Model could help individuals stratification
=======================================================================

This example implement the main results of the following paper:

C Ambroise et al., Multi-modal Latent Variable Model could help individuals
stratification - application to HBN cohort, OHBM 2021.

In recent years, with the rise of population studies, the amount of data
available has constantly increased. A major challenge involves the automatic
stratification of these populations considering the different modalities
available as well as the clinical assessments.

Recently, the Sparse Multi-Channel Variational Autoencoder (sMCVAE) was
proposed. It learns distributions from heterogeneous data in an interpretable
way, exploiting a sparse prior on the learnt distributions.

We propose to train this network on the Healthy Brain Network (HBN) cohort. We
demonstrate the ability of the sMCVAE to robustly identify relevant subgroups
previously defined by a psychiatrist from our group.

Import
------
"""

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from nidl.utils import Bunch, Weights
from nidl.estimators.ae import MCVAE
from nidl.callbacks.caching import CachingCallback

# %%
# Utils
# -----
#
# Define some tools.

class MCDataset(Dataset):
    def __init__(self, X, channel_names=None):
        super().__init__()
        self.channel_names = channel_names
        self.X = [np.expand_dims(_X.astype(np.float32), axis=1) for _X in X]
        sizes = [len(x) for x in self.X]
        self.n_samples = sizes[0]
        assert(all([item == self.n_samples for item in sizes]))
        self.n_feats = [x.shape[2] for x in self.X]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return [x[item] for x in self.X]

    @property
    def shape(self):
        return (len(self), len(self.X))


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
            self.fig.canvas.draw()
            plt.draw()
            plt.pause(self.sleep)


# %%
# Create dataset
# --------------
#
# The Healthy Brain Network cohort brings, among other data, brain MR images,
# and cognitive/behavioral scores in children. It comprises at-risk individuals
# for developing psychiatric disorders and controls [1].
#
# We selected and processed a subset of 584, (train) + 147 (test) individuals.
# Briefly, the T1 structural MRI were processed with FreeSurfer, computing
# thickness, surface area and local gyrification for each cortical region of
# the Destrieux's atlas (444 features). Site effects on cortical features
# were adjusted using the ComBat method. The clinical channel takes as input 7
# behavioral scores used to assess ASD traits: scared_p_total, ari_p_total,
# sdq_hyperactivity, cbcl_ab, cbcl_ap, cbcl_wd, srs_total.
#
# Four data-driven behavioral subgroups were defined by a psychiatrist forming
# our ground truth: emotional dysregulation (Emot), attention problems (Attn),
# anxiety depression (AnxDep) and controls.

with_data = True
if with_data:
    data_params = Bunch(
        channel_names=["clinical", "rois"],
    )
    data = {
        split: np.load(f"./data/multiblock_X_{split}.npz")
        for split in ("train", "test")
    }
    X = {
        split: [
            data[split][key] for key in data_params.channel_names
        ]
        for split in ("train", "test")
    }
    y = {
        split: pd.read_csv(f"./data/metadata_{split}.tsv", sep="\t")
        for split in ("train", "test")
    }
    mapping = dict((key, "Control") for key in range(9))
    mapping[0] = "AnxDep"
    mapping[3] = "Attn"
    mapping[6] = "Emot"
    datasets = {
        split: MCDataset(
            X[split],
            **data_params,
        )
        for split in ("train", "test")
    }
    print(f"-- input train: {[ch.shape for ch in datasets['train'].X]}")
    print(f"-- input validation: {[ch.shape for ch in datasets['test'].X]}")
    dataloaders = {
        split: torch.utils.data.DataLoader(
            datasets[split],
            batch_size=len(datasets[split]),
            shuffle=(split == "train"),
            num_workers=1,
        )
        for split in ["train", "test"]
    }
    n_feats = datasets["train"].n_feats
else:
    n_feats = [7, 444]


# %%
# Training
# --------
#
# Train a sparse MCVAE.
#
# As in many tutorials before, we provide pre-trained models.

load_pretrained = True
checkpointdir = "/tmp/checkpoints"
weight = Weights(
    name="hf-hub:neurospin/mcvae-stratification",
    data_dir=checkpointdir,
    filepath="sMCVAE-stratification.ckpt"
)
if load_pretrained:
    src = weight.weight_file
    dst = os.path.join(checkpointdir, os.path.basename(src))
    if not os.path.exists(dst):
        os.symlink(src, dst) 


model_params = Bunch(
    latent_dim=10,
    beta=1.,
    n_feats=n_feats,
    vae_kwargs={},
    lr=2e-3,
    weight_decay=0,
    max_epochs=10000,
    log_every_n_steps=1,
    random_state=42,
    logger=False,
)
model = MCVAE(
    sparse=True,
    dropout_threshold=0.2,
    callbacks=[
        ModelCheckpoint(
            dirpath="/tmp/checkpoints",
            filename="sMCVAE-stratification",
            save_weights_only=False,
            save_last=False,
            save_top_k=1,
            every_n_epochs=model_params.max_epochs,
            enable_version_counter=False
        ),
        CachingCallback(),
        DropoutRateCallback()
    ],
    **model_params,
)

save_callback, cache_callback = model.trainer_params["callbacks"][:2]
save_path = save_callback.format_checkpoint_name(metrics={})
if not os.path.isfile(save_path):
    model.fit(dataloaders["train"], dataloaders["test"])
    print(f"Saving: {save_callback.last_model_path}...")
    _weights = {
        "model": model.state_dict(),
        "cache": cache_callback.state_dict()
    }
    torch.save(_weights, save_path)
else:
    print(f"Restoring: {save_path}...")
    _weights = torch.load(save_path, weights_only=False)
    model.load_state_dict(_weights["model"])
    cache_callback.load_state_dict(_weights["cache"])
    model.cache = cache_callback.returned_objects
    model.fitted_ = True


# %%
# Display results
# ---------------
#
# # First store latent varaibles, generative parameters, and reconstructed data.

plt.close("all")
q = model.cache.validation[0]
print(f"-- encoded distribution q(z|x): {q}")
z = model.p_to_prediction(q)
print(f"-- z: {[_z.shape for _z in z]}")
z = model.apply_threshold(z, reorder=True)


# %%
# Then, shows the dropout rates for 10 latent dimensions, from which we can
# identify 3 optimal latent dimensions. The encoding of the train and test sets

do = np.sort(model.dropout.numpy().reshape(-1))
plt.figure()
plt.bar(range(len(do)), do)
plt.axhline(
    y=model.hparams.dropout_threshold, color="r", linestyle="--")
plt.gca().spines[["right", "top"]].set_visible(False)
plt.suptitle(
    f"Dropout probability of {model_params.latent_dim} fitted latent "
    "dimensions in sMCVAE",
    fontweight="bold"
)


# %%
# Then, in the latent space given by the sMCVAE are displayed. We can
# see a continuum in the behavioral subgroups indicating that salient clinical
# informations are captured by the sMCVAE latent. This also
# suggests that the sMCVAE is robust to new data.

colors = {
    "Emot": "royalblue",
    "Attn": "goldenrod",
    "AnxDep": "red"
}
if with_data:
    unique_groups = set(mapping.values()) - {"Control"}
    groups = {
        split: [mapping[item] for item in y[split]["labels"]]
        for split in ("train", "test")
    }
    for block, z_block in zip(datasets["test"].channel_names, z):
        print("-- display", block, z_block.shape)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for group in unique_groups:
            group_indices = (np.asarray(groups["test"]) == group)
            _z_block = z_block[group_indices]
            print("-- group", group, _z_block.shape)
            ax.scatter(_z_block[:, 0], _z_block[:, 1], _z_block[:, 2],
                       c=colors[group], marker="o", label=group)
        ax.set_xlabel("1st latent dim")
        ax.set_ylabel("2nd latent dim")
        ax.set_zlabel("3rd latent dim")
        plt.suptitle(
            f"Three selected latent dimensions for the '{block}' channel"
        )
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        plt.legend(loc="upper right")


plt.show()
