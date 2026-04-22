"""
Self-Supervised Learning with I-JEPA on MedMNIST3D
==================================================

This example demonstrates how to pretrain a 3D vision transformer with I-JEPA
[1]_ on a MedMNIST3D dataset [2]_ using the nidl library. It will also show you
how to evaluate the learned representations with a simple linear probe.

**I-JEPA**: the key idea behind I-JEPA is to learn representations by
predicting masked-out image blocks from their surrounding context in the
*latent space* . This is the main difference with Masked Autoencoders (MAE)
which predicts the masked-out blocks in the *pixel space*.

**3D adaptation**: I-JEPA is designed to be flexible. Even if the
original implementation only used 2d images, its extension to 3d volumes
is straightforward. There are two key differences with the 2d case: the
tokenization is performed with 3D patches, and the positional embeddings are
3D as well. As for the masking strategy, it follows the same random block
subsampling strategy as in 2d.

In this tutorial, we will follow these steps:

1. Load a MedMNIST3D dataset.
2. Build a 3D vision transformer encoder.
3. Train an I-JEPA model, or optionally load pretrained weights from the
   Hugging Face Hub.
4. Evaluate the pretrained encoder on the downstream classification task with a
   logistic regression probe.

In this example we use ``OrganMNIST3D``, one of the 3D datasets distributed by
MedMNIST. MedMNIST3D datasets are lightweight 3D medical image classification
benchmarks standardized to a common spatial size, which makes them convenient
for prototyping self-supervised pipelines.

.. [1] Assran, M., Bardes, A., Xu, Q., et al.
       "Self-Supervised Learning from Images with a Joint-Embedding Predictive
       Architecture", CVPR 2023.
.. [2] Yang, J., et al."MedMNIST v2 - A large-scale lightweight benchmark for
       2D and 3D biomedical image classification", Nature Scientific Data 2023.

Setup
-----
This example requires ``medmnist`` in addition to ``nidl``. If you want to load
pretrained weights from the Hugging Face Hub, you also need
``huggingface_hub`` installed in your environment.
"""

# %%
from __future__ import annotations

import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import medmnist
import numpy as np
import torch
from lightning_fabric import seed_everything
from medmnist import INFO
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from nidl.estimators.ssl import IJEPA
from nidl.utils.weights import Weights
from nidl.volume.backbones import VisionTransformer3D
from nidl.volume.transforms.augmentation import RandomResizedCrop
from nidl.volume.transforms.preprocessing import ZNormalization

# %%
# We define some global parameters that will be used throughout the example.
#
# Training a 3D I-JEPA model can take substantial time depending on your
# hardware. By default, this example trains a lightweight configuration that is
# suitable for a tutorial. If you later publish pretrained weights to the
# Hugging Face Hub, set ``load_pretrained = True`` and fill the corresponding
# repository information below.

# %%
# Data-related parameters

# Directory where to download MedMNIST data
data_dir = "/tmp/medmnist"
# Directory where to cache optional pretrained weights
model_dir = "/tmp/nidl_example_ijepa_medmnist"
# MedMNIST3D dataset to use
dataset_name = "organmnist3d"
# Spatial size used by MedMNIST+ for 3D datasets
img_size = 64
# Whether to load a pretrained checkpoint from HF or train locally
load_pretrained = True
# Fill these two values once a checkpoint is published on HF
hf_repo_id = "neurospin/nidl_example_ijepa_medmnist"
hf_checkpoint = "nidl_example_ijepa_medmnist.ckpt"

# %%
# Reproducibility and training configuration

# What accelerator to use: GPU if available, else CPU
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
# Parameters for the data loaders. Reduce them if you run out of memory.
batch_size = 16
num_workers = 4
# Training configuration
max_epochs = 20
learning_rate = 3e-4
weight_decay = 5e-4
random_seed = 42

seed_everything(random_seed)
rd_generator = np.random.default_rng(seed=random_seed)

# %%
# Data preparation
# ----------------
#
# We first define a small MedMNIST3D dataset wrapper and the transforms used
# for self-supervised pretraining and downstream evaluation.


# %%
class MedMNIST3DDataset:
    """Simple wrapper around a MedMNIST3D split."""

    def __init__(
        self,
        dataset_name: str,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        size: int = 64,
        download: bool = True,
    ):
        dataset_name = dataset_name.lower()
        if dataset_name not in INFO:
            raise ValueError(
                f"Unknown MedMNIST dataset '{dataset_name}'. "
                f"Available datasets include: {sorted(INFO.keys())}"
            )
        if "3d" not in dataset_name:
            raise ValueError(
                f"This example is written for MedMNIST3D datasets, got "
                f"'{dataset_name}'."
            )

        info = INFO[dataset_name]
        dataset_cls = getattr(medmnist, info["python_class"])
        os.makedirs(root, exist_ok=True)
        self.dataset = dataset_cls(
            split=split,
            root=root,
            transform=transform,
            download=download,
            size=size,
            as_rgb=False,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


# %%
train_transform = Compose(
    [
        RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        ZNormalization(),
        lambda x: torch.from_numpy(x).float(),
    ]
)

eval_transform = Compose(
    [
        ZNormalization(),
        lambda x: torch.from_numpy(x).float(),
    ]
)

# %%
# We create one dataset for self-supervised training, and labeled datasets for
# linear probing.

# %%
ssl_dataset = MedMNIST3DDataset(
    dataset_name=dataset_name,
    root=data_dir,
    split="train",
    transform=train_transform,
    size=img_size,
)

train_xy_dataset = MedMNIST3DDataset(
    dataset_name=dataset_name,
    root=data_dir,
    split="train",
    transform=eval_transform,
    size=img_size,
)

test_xy_dataset = MedMNIST3DDataset(
    dataset_name=dataset_name,
    root=data_dir,
    split="test",
    transform=eval_transform,
    size=img_size,
)

# %%
# Finally, we create the data loaders.

# %%
train_ssl_loader = DataLoader(
    ssl_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    num_workers=num_workers,
)
train_xy_loader = DataLoader(
    train_xy_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    num_workers=num_workers,
)
test_xy_loader = DataLoader(
    test_xy_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    num_workers=num_workers,
)

# %%
# Model architecture
# ------------------
#
# We use the 3D vision transformer from `nidl` as backbone.

# %%
encoder = VisionTransformer3D(
    img_size=img_size,
    patch_size=8,
    in_chans=1,
    embed_dim=256,
    depth=6,
    num_heads=8,
    mlp_ratio=4.0,
)

# %%
# Training the I-JEPA model
# -------------------------
#
# We either train a compact I-JEPA model directly, or load a pretrained
# checkpoint from the Hugging Face Hub (which was trained with the same
# configuration).

# %%
if not load_pretrained:
    model = IJEPA(
        encoder=encoder,
        dim=3,
        context_block_scale=(0.85, 1.0),
        target_block_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        num_target_blocks=4,
        min_keep=4,
        allow_overlap=False,
        predictor_embed_dim=256,
        predictor_depth_pred=6,
        learning_rate=learning_rate,
        optimizer="adamW",
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        use_distributed_sampler=False,
        enable_checkpointing=False,
        accelerator=accelerator,
        devices=1,
        random_state=random_seed,
    )

    model.fit(train_ssl_loader)

# %%
if load_pretrained:
    weights = Weights(
        f"hf-hub:{hf_repo_id}",
        data_dir=model_dir,
        filepath=hf_checkpoint,
    )
    model = weights.load_checkpoint(
        IJEPA,
        encoder=encoder,
        devices=1,
        accelerator=accelerator,
        enable_checkpointing=False,
        logger=False,
    )

# %%
# Evaluation with a linear probe
# ------------------------------
#
# Once pretraining is complete, we extract frozen representations and fit a
# logistic regression classifier on top of them.

# %%
X_train, y_train = model.transform_with_targets(train_xy_loader)
X_test, y_test = model.transform_with_targets(test_xy_loader)

X_train = X_train.cpu().numpy()
y_train = y_train.cpu().numpy().reshape(-1)
X_test = X_test.cpu().numpy()
y_test = y_test.cpu().numpy().reshape(-1)

# %%
# We train linear probes on increasing fractions of the labeled training set to
# assess sample efficiency.

# %%
estimator = LogisticRegression(
    max_iter=1000, random_state=random_seed, n_jobs=1
)
train_sizes = np.unique(
    np.logspace(
        np.log10(max(10, len(X_train) // 20)),
        np.log10(len(X_train)),
        8,
        dtype=int,
    )
)
accs = []
for size in train_sizes:
    estimator.fit(X_train[:size], y_train[:size])
    y_pred = estimator.predict(X_test)
    accs.append(accuracy_score(y_test, y_pred))

# %%
# We plot the scaling curve.

# %%
plt.plot(train_sizes / len(X_train), accs)
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.xlabel("Proportion of labeled training samples")
plt.xscale("log")
plt.text(
    train_sizes[-1] / len(X_train),
    accs[-1],
    f"{accs[-1]:.2f}",
    ha="right",
    va="bottom",
)
plt.text(
    train_sizes[0] / len(X_train),
    accs[0],
    f"{accs[0]:.2f}",
    ha="left",
    va="bottom",
)
plt.show()

# %%
# This example shows how to train and evaluate the I-JEPA model on MedMNIST3D
# using ``nidl``. The same pipeline can be applied to other 3D medical imaging
# datasets, and you can also expand it to include more complex training setups
# with logging and callbacks.
