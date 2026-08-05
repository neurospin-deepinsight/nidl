"""
Surfacic backbones
==================

This example illustrates how to work with *surfacic backbones* in the
``nidl`` library, focusing on SCNN‑style architectures for cortical
surface data.

We show how to:

- load a pretrained **HemiSCNN** model trained with SimCLR on OpenBHB,
- prepare left/right hemispheric metrics from the ``fs_xhemi`` modality,
- downsample high‑resolution icospheres to the resolution expected by the model,
- and finally compute **latent representations** for a single subject.

Surfacic backbones are particularly useful when working with cortical
geometry, as they operate directly on meshes rather than volumetric grids.


Imports
-------

We import the surface utilities, the pretrained backbone, and the OpenBHB
dataset.
"""

from pathlib import Path
import numpy as np
import torch

from nidl.surface import Ico
from nidl.surface.backbones import (
    HemiSCNN,
    SiT,
    SiTAttentionMaps,
)
from nidl.datasets import OpenBHB


# %%
# Load a pretrained HemiSCNN model
# --------------------------------
#
# The pretrained model was trained using SimCLR on the OpenBHB dataset.
# It expects a pair of hemispheres (left, right), each containing 3 metrics:
# thickness, curvature, and sulcal depth.

scnn_model = HemiSCNN.from_pretrained(
    name="openbhb_simclr_v1",
    weights_file=Path(__file__).parent / "weights-openbhb.pt",
    printer=print,
    cache_file="/tmp/cache.npz",
    n_jobs=30,
)
print(scnn_model)


# %%
# Load a pretrained SiT model
# ---------------------------
#
# The pretrained model was trained on ImageNet dataset.
# It expects a pair of hemispheres (left, right), each containing 3 metrics:
# thickness, curvature, and sulcal depth.
# Note that in the SiT original paper they use sulcal depth, curvature,
# cortical thickness and T1w/T2w myelination modalities as inputs.


sit_model = SiT.from_pretrained(
    name="vit_tiny",
    n_channels=3,
    n_classes=0,
    printer=print,
    cache_file="/tmp/cache_sit.npz",
    n_jobs=30,
)
print(sit_model)


# %%
# Load an OpenBHB subject
# -----------------------
#
# We load one subject from the validation split. The ``fs_xhemi`` modality
# provides cortical metrics mapped to a symmetric icosphere.

dataset = OpenBHB(
    root="/tmp/openBHB",
    modality=(
        "fs_xhemi",
    ),
    target=["age", "sex", "site"],
    split="val",
)

xhemi_data, infos = dataset[0]

# %%
# Downsample the surface
# ----------------------
#
# The pretrained model expects data on an icosphere of order 5 or 6.
# The OpenBHB fs_xhemi data is provided at order 7, so we downsample it.

down7to5_indices = Ico.downsample(
    source_order=7,
    target_order=5,
    standard_ico=False,
)
down7to6_indices = Ico.downsample(
    source_order=7,
    target_order=6,
    standard_ico=False,
)

# %%
# Prepare hemispheric inputs
# --------------------------
#
# We extract the three metrics for each hemisphere and apply the downsampling.
# The resulting tensors have shape (3, n_vertices), where n_vertices
# corresponds to the order‑5 or order-6 icosphere.

data5, data6 = {}, {}
metric_indices = dataset.get_fs_xhemi_feature_names()

for hemi in ("lh", "rh"):
    data = np.array([
        xhemi_data[metric_indices.index(f"{hemi}.thickness")],
        xhemi_data[metric_indices.index(f"{hemi}.curv")],
        xhemi_data[metric_indices.index(f"{hemi}.sulc")],
    ])
    mask = data.sum(axis=0) > 0
    print(mask.shape)
    data[:, mask] = (
        (data[:, mask] - data[:, mask].mean(axis=1, keepdims=True)) /
        data[:, mask].std(axis=1, keepdims=True)
    )
    data5[hemi] = torch.from_numpy(
        data[:, down7to5_indices],
    )
    data6[hemi] = torch.from_numpy(
        data[:, down7to6_indices],
    )

print("Data ico 5:", [(key, arr.shape) for key, arr in data5.items()])
print("Data ico 6:",[(key, arr.shape) for key, arr in data6.items()])


# %%
# Compute latent embeddings
# -------------------------
#
# The models outputs a 128‑ or 192-dimensional latent representation
# summarizing the geometry and local patterns of both hemispheres.

z = scnn_model((data5["lh"][None, ...], data5["rh"][None, ...]))
print("Z(SCNN):", z.shape)

z = sit_model(torch.cat((data6["lh"][None, ...], data6["rh"][None, ...])))
print("Z(SiT):", z.shape)

# %%
# The resulting embedding ``z`` can now be used for downstream tasks such as:
# - similarity analysis
# - clustering
# - regression/classification


# %%
# Attention maps
# --------------
#
# Attention maps per head for a subjects using the SiT model.


att_model = SiTAttentionMaps(
    model=sit_model,
)
left_maps = att_model(data6["lh"][None, ...]).mean(axis=1)
right_maps = att_model(data6["rh"][None, ...]).mean(axis=1)
print("Attention maps (SiT):", left_maps.shape, right_maps.shape)
