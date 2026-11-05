"""
Pretraining recipe of the open models
=====================================

The open models of nidl are pretrained with nidl only, on open data:
OpenMind [1]_, 787 OpenNeuro datasets of T1-weighted, T2-weighted and FLAIR
brain MRI. Their data, code, training recipe and weights are all public, so
that anyone can reproduce them, compare a new method with them under the
same conditions, or build upon them.

This tutorial is their pretraining recipe. It reproduces how the weights
published on the `nidl-neurospin <https://huggingface.co/nidl-neurospin>`_
organization were obtained, for each of the three open models:

- ``open-simclr-resnet50``: SimCLR [2]_ with a 3D ResNet-50;
- ``open-barlowtwins-resnet50``: Barlow Twins [3]_ with a 3D ResNet-50;
- ``open-neurojepa-vit-b``: NeuroJEPA [4]_ with a 3D ViT-B with mixtures of
  experts.

The recipe has two parts:

1. **the estimator**: its architecture, hyper-parameters (optimizer,
   learning rate schedule, loss...) and training parameters (number of
   epochs, precision...) are all in the ``config.json`` of the model, from
   which ``from_config`` builds the same estimator, randomly initialized;
2. **the data**: the images, their augmentations and their loading, given
   below.

.. note::
    Each model was trained for 100 epochs on one NVIDIA H100 GPU, in 32 to
    47 hours (validation included). The quasi-raw OpenMind images take about
    460 GB of disk, downloaded on first use. For these reasons, this example
    is not run when building the documentation: set ``max_steps`` to try it
    on a few batches.

.. [1] Wald et al., "An OpenMind for 3D medical vision self-supervised
       learning", ICCV, 2025.
.. [2] Chen et al., "A Simple Framework for Contrastive Learning of Visual
       Representations", ICML, 2020.
.. [3] Zbontar et al., "Barlow Twins: Self-Supervised Learning via
       Redundancy Reduction", ICML, 2021.
.. [4] Huang et al., "Learning Sparse Latent Predictive Foundation Model for
       Multimodal Neuroimaging", arXiv:2606.14957, 2026.

Setup
-----

Let's first import the libraries used in this tutorial:
"""

import json

from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from nidl.datasets import OpenMind
from nidl.estimators import ssl as estimators
from nidl.transforms import MultiViewsTransform
from nidl.transforms.volume.augmentation import (
    RandomErasing,
    RandomGaussianNoise,
    RandomResizedCrop,
)
from nidl.transforms.volume.preprocessing import CropOrPad, ZNormalization

# %%
# We define some global parameters: the model to pretrain, the OpenMind
# folder (downloaded on first use), the loading options and the number of
# training steps (None for the whole training, as published).
repo_id = "nidl-neurospin/open-simclr-resnet50"
data_dir = "/tmp/openmind"
num_workers = 20
max_steps = None

# %%
# The estimator
# -------------
#
# The ``config.json`` of the model describes the estimator entirely. Let's
# look at the hyper-parameters of its training:

config_file = hf_hub_download(repo_id, "config.json")
with open(config_file) as f:
    config = json.load(f)
print(config["estimator"])
print(json.dumps(config["estimator_kwargs"], indent=2))
print(config["trainer"])

# %%
# ``from_config`` builds this estimator, with random weights. The trainer
# section of the configuration (number of epochs, mixed precision, gradient
# clipping) is applied too, since the learning rate schedule depends on the
# number of epochs. Only the hardware is ours to choose: here, one GPU.

Estimator = getattr(estimators, config["estimator"])
trainer_params = {"devices": 1}
if max_steps is not None:
    trainer_params["max_steps"] = max_steps
model = Estimator.from_config(config, **trainer_params)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

# %%
# The data
# --------
#
# The open models are pretrained on the train split of OpenMind and
# validated on its val split, split by subject. All the quasi-raw images are
# used, whatever their modality (T1w, T2w and FLAIR): 63,992 images of 31,775
# subjects for training and 7,498 for validation. They are skull-stripped,
# bias-corrected and linearly registered to MNI152 at 1 mm, of shape
# :math:`182 \times 218 \times 182`.
#
# The augmentations depend on the method. SimCLR and Barlow Twins compare
# two views of each image, made of random erasing, a random resized crop to
# :math:`121 \times 145 \times 121`, a center crop or padding to
# :math:`128^3`, a standardization and some Gaussian noise:

contrastive_augmentations = MultiViewsTransform(
    Compose(
        [
            RandomErasing(scale=(0.02, 0.15), p=0.8),
            RandomResizedCrop(
                target_shape=(121, 145, 121), scale=(0.08, 1.0), p=0.8
            ),
            CropOrPad(target_shape=128),
            ZNormalization(),
            RandomGaussianNoise(p=0.2),
        ]
    ),
    n_views=2,
)

# %%
# NeuroJEPA predicts masked regions of a single view from the visible ones:
# the masks are drawn by the estimator, and the augmentation is only a
# milder random resized crop.

neurojepa_augmentations = Compose(
    [
        RandomResizedCrop(
            target_shape=(121, 145, 121), scale=(0.6, 1.0), p=0.8
        ),
        CropOrPad(target_shape=128),
        ZNormalization(),
    ]
)

augmentations = {
    "SimCLR": contrastive_augmentations,
    "BarlowTwins": contrastive_augmentations,
    "NeuroJEPA": neurojepa_augmentations,
}[config["estimator"]]

# %%
# The validation images get the same augmentations, so that the validation
# loss is comparable to the training one. Batches hold 64 images, and
# incomplete batches are dropped: one epoch is 999 training steps.

datasets = {
    split: OpenMind(
        data_dir,
        preprocessing="quasiraw",
        split=split,
        transforms=augmentations,
    )
    for split in ("train", "val")
}
train_loader = DataLoader(
    datasets["train"],
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
    persistent_workers=num_workers > 0,
)
val_loader = DataLoader(
    datasets["val"],
    batch_size=64,
    drop_last=True,
    num_workers=num_workers,
)
print(
    f"{len(datasets['train'])} training and {len(datasets['val'])} "
    "validation images"
)

# %%
# Pretraining
# -----------
#
# The estimator is then trained as any nidl estimator:

model.fit(train_loader, val_loader)

# %%
# Sharing the model
# -----------------
#
# The trained estimator is saved in the same format as the published one.
# It keeps the configuration it was built from, including the preprocessing
# its features expect at inference: the whole image is resized to
# :math:`121 \times 145 \times 121`, as by the crops of the training, then
# center cropped or padded and standardized. It can also be published
# directly with ``push_to_hub``.

print(config["preprocessing"])
model.save_pretrained("my-open-model")

# %%
# Conclusion
# ----------
#
# The weights obtained this way are trained exactly as the published ones:
# same data, same code, same hyper-parameters. They are not identical bit for
# bit, since the published runs were not seeded and GPU training is not
# deterministic: pass ``random_state`` to ``from_config`` to seed yours.
# Their features can be compared to the published ones, e.g., on brain age
# with :ref:`sphx_glr_auto_examples_plot_pretrained_openbhb.py`, and a new
# method can be trained with the same recipe to be compared with the open
# models under the same conditions.
