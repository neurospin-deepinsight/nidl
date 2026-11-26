"""
Brain age with pretrained models
================================

Several self-supervised models have been pretrained on large cohorts of brain
MRI and their weights published. nidl loads them with their exact
architecture and preprocessing, so that they can be used directly in a
pipeline, as frozen feature extractors or as a starting point for
fine-tuning.

This tutorial compares five of them on brain age prediction on OpenBHB [1]_:

- y-Aware [2]_, a DenseNet-121 pretrained on gray matter maps (VBM);
- Neuro-SimCLR [3]_, a ResNet-18 pretrained on T1-weighted images;
- BrainIAC [4]_, a ViT-B pretrained on T1-weighted images;
- Neuro-JEPA [5]_, ViT-B and ViT-L with sparse mixtures of experts,
  pretrained on multimodal brain MRI.

For each model, we will:

1. load its weights with ``from_pretrained``;
2. preprocess the OpenBHB images as the model expects, with the
   preprocessing published with it, loaded by ``Transform.from_pretrained``;
3. extract frozen features of the training and test images;
4. fit a ridge regression of age on these features and compare the true and
   predicted ages on the test set;
5. look at the main axes of variation of the features (PCA), colored by age
   and sex.

.. note::
    The Neuro-JEPA weights are gated: request access on their `model page
    <https://huggingface.co/NYUMedML/Neuro-JEPA>`_, then authenticate with
    ``huggingface-cli login``. The other weights are hosted by the
    `nidl-neurospin <https://huggingface.co/nidl-neurospin>`_ organization.
    Extracting the features of the whole OpenBHB dataset with all the models
    takes about an hour on one GPU: set ``max_samples`` to try it on fewer
    images. For this reason, this example is not run when building the
    documentation: its results on the whole dataset are shown at the end.

.. [1] Dufumier et al., "OpenBHB: a Large-Scale Multi-Site Brain MRI
       Data-set for Age Prediction and Debiasing", NeuroImage, 2022.
.. [2] Dufumier et al., "Contrastive Learning with Continuous Proxy Meta-data
       for 3D MRI Classification", MICCAI, 2021.
.. [3] Kaczmarek et al., "Building a General SimCLR Self-Supervised
       Foundation Model Across Neurological Diseases to Advance 3D Brain MRI
       Diagnoses", ICCV Workshops, 2025.
.. [4] Tak et al., "A generalizable foundation model for analysis of human
       brain MRI", Nature Neuroscience, 2026.
.. [5] Huang et al., "Learning Sparse Latent Predictive Foundation Model for
       Multimodal Neuroimaging", arXiv:2606.14957, 2026.

Setup
-----

Let's first import the libraries used in this tutorial:
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from nidl.datasets import OpenBHB
from nidl.estimators.ssl import NeuroJEPA, SimCLR, YAwareContrastiveLearning
from nidl.transforms import Transform

# %%
# We define some global parameters: the OpenBHB folder (downloaded on first
# use), the number of images per split (None for all of them) and the loading
# options.
data_dir = "/tmp/openBHB"
max_samples = None
batch_size = 8
num_workers = 8

# %%
# Pretrained models
# -----------------
#
# Each model is described by its estimator class, the Hugging Face repository
# holding its weights and its preprocessing, the OpenBHB modality it takes,
# and ``preprocessing_kwargs``: the parameters of its preprocessing that are
# specific to OpenBHB. They are given to ``Transform.from_pretrained``, and
# complete, or replace, the parameters saved with the model.
#
# Arrays do not carry their orientation: OpenBHB images are in the LAS
# orientation of their template (the first axis goes from right to left).
# Two models were also trained on volumes registered to another template
# than OpenBHB's: the images are brought to the field of view of these
# templates from their world coordinates, without interpolation, with a crop
# for BrainIAC and a zero-padding for Neuro-SimCLR. The y-Aware model was
# trained on the same VBM preprocessing as OpenBHB.

model_configs = {
    "y-Aware (VBM)": {
        "estimator": YAwareContrastiveLearning,
        "repo_id": "nidl-neurospin/yaware-densenet121-bhb10k",
        "modality": "vbm",
        "preprocessing_kwargs": {},
    },
    "Neuro-SimCLR": {
        "estimator": SimCLR,
        "repo_id": "nidl-neurospin/neuro-simclr-resnet18",
        "modality": "quasiraw",
        "preprocessing_kwargs": {
            "input_orientation": "LAS",
            "padding": ((5, 6), (6, 5), (6, 5)),
        },
    },
    "BrainIAC": {
        "estimator": SimCLR,
        "repo_id": "nidl-neurospin/brainiac-vit-b",
        "modality": "quasiraw",
        "preprocessing_kwargs": {"crop": ((6, 176), (7, 213), (3, 165))},
    },
    "Neuro-JEPA": {
        "estimator": NeuroJEPA,
        "repo_id": "nidl-neurospin/neuro-jepa-vit-b",
        "modality": "quasiraw",
        "preprocessing_kwargs": {"input_orientation": "LAS"},
    },
    "Neuro-JEPA-Large": {
        "estimator": NeuroJEPA,
        "repo_id": "nidl-neurospin/neuro-jepa-vit-l",
        "modality": "quasiraw",
        "preprocessing_kwargs": {"input_orientation": "LAS"},
    },
}

# %%
# Loading a pretrained model is a single call. The returned estimator holds
# the architecture and the weights of the published model. The extra
# parameters configure the PyTorch Lightning trainer used to extract the
# features. The preprocessing of its inputs is published in the same
# repository: ``Transform.from_pretrained`` returns the transform reproducing
# the preprocessing of its original implementation, given the parameters that
# depend on our data.

model = SimCLR.from_pretrained(
    "nidl-neurospin/brainiac-vit-b", devices=1, enable_progress_bar=False
)
transform = Transform.from_pretrained(
    "nidl-neurospin/brainiac-vit-b",
    **model_configs["BrainIAC"]["preprocessing_kwargs"],
)
print(transform.get_config())

# %%
# Feature extraction
# ------------------
#
# We extract the features of the OpenBHB training and test sets (the official
# test set is the public ``"val"`` split). Each image comes with the age and
# sex of the participant, encoded as a tensor ``[age, is_female]``.


def encode_targets(target):
    return torch.tensor(
        [target["age"], target["sex"] == "female"], dtype=torch.float32
    )


def extract_features(model, config, split):
    dataset = OpenBHB(
        data_dir,
        modality=config["modality"],
        target=["age", "sex"],
        split=split,
        transforms=Transform.from_pretrained(
            config["repo_id"], **config["preprocessing_kwargs"]
        ),
        target_transforms=encode_targets,
    )
    if max_samples is not None:
        rng = np.random.default_rng(0)
        indices = rng.permutation(len(dataset))[:max_samples]
        dataset = Subset(dataset, np.sort(indices))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    features, targets = model.transform_with_targets(loader)
    return features.float().cpu().numpy(), targets.cpu().numpy()


# %%
# The Neuro-JEPA models are run in bfloat16, as in their original code. The
# others are run in full precision.

features = {}
for name, config in model_configs.items():
    precision = "bf16-mixed" if config["estimator"] is NeuroJEPA else None
    model = config["estimator"].from_pretrained(
        config["repo_id"],
        devices=1,
        precision=precision,
        enable_progress_bar=False,
        logger=False,
    )
    features[name] = {
        split: extract_features(model, config, split)
        for split in ("train", "val")
    }
    print(f"{name}: {features[name]['train'][0].shape[1]} features")
    del model
    torch.cuda.empty_cache()

# %%
# Brain age prediction
# --------------------
#
# A ridge regression, with its regularization chosen by cross-validation on
# the training set, predicts the age from the standardized features. The
# encoders stay frozen: this measures how linearly the age can be read from
# the pretrained representations.

predictions, scores = {}, []
for name, splits in features.items():
    (X_train, y_train), (X_test, y_test) = splits["train"], splits["val"]
    regressor = make_pipeline(
        StandardScaler(), RidgeCV(alphas=np.logspace(-2, 6, 17))
    )
    regressor.fit(X_train, y_train[:, 0])
    predictions[name] = y_pred = regressor.predict(X_test)
    scores.append(
        {
            "model": name,
            "features": X_train.shape[1],
            "MAE (years)": mean_absolute_error(y_test[:, 0], y_pred),
            "R2": r2_score(y_test[:, 0], y_pred),
        }
    )
scores = pd.DataFrame(scores).set_index("model")
print(scores.round(3))

# %%
# We compare the true and predicted ages of the test participants, the
# diagonal standing for a perfect prediction. Ridge regressions shrink their
# predictions toward the mean age, so that young participants tend to be
# predicted older and old ones younger.

surface, ink, muted = "#fcfcfb", "#0b0b0b", "#52514e"
blue, orange = "#2a78d6", "#eb6834"
plt.rcParams.update(
    {
        "figure.facecolor": surface,
        "axes.facecolor": surface,
        "axes.edgecolor": muted,
        "axes.labelcolor": ink,
        "xtick.color": muted,
        "ytick.color": muted,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

y_test = next(iter(features.values()))["val"][1]
ages = y_test[:, 0]
age_range = (ages.min() - 2, ages.max() + 2)
fig, axes = plt.subplots(
    1,
    len(model_configs),
    figsize=(3.2 * len(model_configs), 3.6),
    sharex=True,
    sharey=True,
)
for ax, name in zip(axes, model_configs, strict=True):
    ax.plot(age_range, age_range, color=muted, linewidth=1, linestyle="--")
    ax.scatter(
        ages,
        predictions[name],
        s=10,
        color=blue,
        alpha=0.6,
        edgecolors=surface,
        linewidths=0.3,
    )
    ax.set_title(name, color=ink, fontsize=10)
    ax.text(
        0.04,
        0.96,
        f"MAE {scores.loc[name, 'MAE (years)']:.2f} y\n"
        f"R² {scores.loc[name, 'R2']:.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color=ink,
    )
    ax.set_xlabel("True age (years)")
    ax.set_xlim(age_range)
    ax.set_ylim(age_range)
    ax.set_aspect("equal")
    ax.grid(color="#e8e7e3", linewidth=0.5)
axes[0].set_ylabel("Predicted age (years)")
fig.suptitle(
    "Brain age on the OpenBHB test set, ridge on frozen features",
    color=ink,
)
fig.tight_layout()
plt.show()

# %%
# Main axes of variation of the features
# --------------------------------------
#
# Without any label, how do the representations of the test images organize?
# We project the standardized features on their first two principal
# components, and color the participants by age (top) and sex (bottom). Sex
# is also shown by the marker shape, so that it does not rely on color only.
# The fraction of variance explained by each component is given on its axis.

age_colors = LinearSegmentedColormap.from_list(
    "age", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"]
)
is_female = y_test[:, 1].astype(bool)
fig, axes = plt.subplots(
    2,
    len(model_configs),
    figsize=(3.2 * len(model_configs), 6.8),
    squeeze=False,
    layout="constrained",
)
for column, name in enumerate(model_configs):
    X_test = StandardScaler().fit_transform(features[name]["val"][0])
    pca = PCA(n_components=2, random_state=0)
    components = pca.fit_transform(X_test)
    variance = pca.explained_variance_ratio_ * 100
    age_ax, sex_ax = axes[:, column]
    points = age_ax.scatter(
        *components.T,
        c=ages,
        cmap=age_colors,
        s=10,
        edgecolors=surface,
        linewidths=0.3,
    )
    for mask, label, color, marker in (
        (~is_female, "Male", blue, "o"),
        (is_female, "Female", orange, "^"),
    ):
        sex_ax.scatter(
            *components[mask].T,
            s=12,
            color=color,
            marker=marker,
            alpha=0.6,
            edgecolors=surface,
            linewidths=0.3,
            label=label,
        )
    age_ax.set_title(name, color=ink, fontsize=10)
    for ax in (age_ax, sex_ax):
        ax.set_xlabel(f"PC 1 ({variance[0]:.0f}%)")
        ax.set_ylabel(f"PC 2 ({variance[1]:.0f}%)")
        ax.set_xticks([])
        ax.set_yticks([])
fig.colorbar(points, ax=axes[0, -1], label="Age (years)")
fig.colorbar(points, ax=axes[1, -1]).remove()  # keeps the columns aligned
handles, labels = axes[1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
fig.suptitle("PCA of the frozen features of the test images", color=ink)
plt.show()

# %%
# Results on the whole dataset
# ----------------------------
#
# The figures obtained on the whole OpenBHB dataset, with one GPU:
#
# .. image:: /images/plot_pretrained_openbhb_age.png
#    :alt: True and predicted ages of the OpenBHB test participants, for each
#          pretrained model.
#
# .. image:: /images/plot_pretrained_openbhb_pca.png
#    :alt: First two principal components of the features of each model,
#          colored by age and by sex.
#
# Conclusion
# ----------
#
# In a few lines, nidl gives access to published models with the exact
# architecture and preprocessing of their original implementation. Their
# frozen features already predict the brain age accurately with a simple ridge
# regression.
#
# The same estimators can be fine-tuned with ``fit``, saved with
# ``save_pretrained`` and shared with ``push_to_hub``.
