"""
Spherical augmentations
=======================

This example shows how to build and visualize spherical augmentations: cutout,
rotation, Gaussian noise, and Gaussian blur.

Build an icosahedral sphere
---------------------------

We start from creating an icosahedron of a specific order. Higher order means
more vertices and triangles on the sphere. We also create a synthetic texture
composed of ones.
"""

import numpy as np
from nidl.surface.base import Ico

kwargs = {
    "order": 3,
    "standard_ico": False,
}

verts, tris = Ico.mesh(**kwargs)
texture = np.array([1, ] * len(verts))
print(f"Mesh: {verts.shape} - {tris.shape}")
print(f"Texture: {texture.shape}")

# %%
# This gives us a spherical mesh: a discretized sphere built from an
# icosahedron.
#
# Cutout
# ------
#
# Cutout is a form of occlusion augmentation. On an icosahedral mesh, it
# randomly removes region of the texture defined as a vetex neighbors, forcing
# the model to rely on global structure rather than memorizing local details.
# The cutout strength is controlled by a parameter called size.
# Increasing size makes the cutout wider.
# Cutout improves robustness to missing data, local corruption and prevents the
# model for over‑relying on specific high‑frequency regions of the sphere.

from nidl.surface.transforms import RandomCutOut

max_size = Ico.min_rings(np.ceil(len(verts) // 16))
print(f"- max estimated size: {max_size}")
transform = RandomCutOut(
    cuts=(1, 3),
    size=(2, max_size),
    standard_ico=kwargs["standard_ico"],
)
cutout_data = [
    transform(texture) for _ in range(4)
]
print(f"Cutout textures: {[text.shape for text in cutout_data]}")

# Rotation
# --------
#
# A rotation in 3D space is applied to the sphere, and the texture is
# re‑sampled at the rotated vertex positions. This produces a texture that is
# physically rotated on the sphere.
# Rotation augmentation teaches the model to be invariant to small
# misalignments.

from nidl.surface.transforms import RandomRotation

neighs = Ico.neighbors(depth=1, **kwargs)
transform = RandomRotation(
    phi=(5., 180.),
    theta=0.0,
    psi=0.0,
    standard_ico=kwargs["standard_ico"],
)
texture_ = texture.copy()
texture_[neighs[0]] = 0
rotation_data = [
    transform(texture_) for _ in range(4)
]
print(f"Rotation textures: {[text.shape for text in rotation_data]}")

# Gaussian noise
# --------------
#
# Noise augmentation introduces small random variations into the texture. This
# forces the model to learn stable, meaningful patterns rather than memorizing
# exact vertice values.
# We adds small random fluctuations drawn from a normal distribution.
# This augmentation is suseful for simulating natural variability.

from nidl.surface.transforms import RandomGaussianNoise

transform = RandomGaussianNoise(mean=0, sigma=(0.01, 0.1))
noisy_data = [
    transform(texture) for _ in range(4)
]
print(f"Noisy textures: {[text.shape for text in noisy_data]}")

# Guassian blur
# -------------
#
# Gaussian blur on an icosahedral mesh works by assigning a weight to each
# ring around a vertex. The weight decreases with distance from the center,
# following a Gaussian curve. The blur strength is controlled by a parameter
# called sigma.
# Increasing sigma makes the blur wider and softer, while decreasing sigma
# keeps more detail.
# This produces a smooth, isotropic blur that respects the spherical geometry.
# Here, we add a zero hole in the texture to see the bluring effect.

from nidl.surface.transforms import RandomGaussianBlur

neighs = Ico.neighbors(depth=2, **kwargs)
transform = RandomGaussianBlur(
    sigma=(0.0, 1.5),
    standard_ico=kwargs["standard_ico"],
)
texture_ = texture.copy()
texture_[neighs[0]] = 0
blured_data = [
    transform(texture_) for _ in range(4)
]
print(f"Blured textures: {[text.shape for text in blured_data]}")

# Combining augmentations
# -----------------------
#
# All previous augmentations complement each other. Together, they produce
# a model that is better at generalizing across individual variations.

from torchvision import transforms

transform = transforms.Compose([
    RandomGaussianNoise(
        mean=0,
        sigma=(0.01, 0.1),
    ),
    transforms.RandomApply([
       RandomCutOut(
            cuts=(1, 3),
            size=(2, max_size),
            standard_ico=kwargs["standard_ico"],
        ),
       RandomRotation(
            phi=(5., 180.),
            theta=0.0,
            psi=0.0,
            standard_ico=kwargs["standard_ico"],
        ),
       RandomGaussianNoise(
            mean=0,
            sigma=(0.01, 0.1),
        ),
       RandomGaussianBlur(
            sigma=(0.0, 1.5),
            standard_ico=kwargs["standard_ico"],
        ),
    ], p=0.5),
    
])
augmented_data = [
    transform(texture.copy()) for _ in range(4)
]
print(f"Augmented textures: {[text.shape for text in blured_data]}")


# Visualize the mesh and neighbors
# --------------------------------
#
# We visualize the generated augmentations.

import matplotlib.pyplot as plt
from nidl.surface.plotting import IcoRenderer

for data in (
        cutout_data,
        rotation_data,
        noisy_data,
        blured_data,
        augmented_data):
    fig, axs = plt.subplots(
        2, 2,
        subplot_kw={"projection": "3d", "aspect": "auto"},
        figsize=(10, 10),
    )
    axs = axs.flatten()
    for ax, texture in zip(axs, data, strict=True):
        ren = IcoRenderer(
            fig=fig,
            ax=ax,
        )
        ren.add_trisurf(
            verts,
            tris,
        )
        ren.add_texture(
            texture,
        )
ren.show()
