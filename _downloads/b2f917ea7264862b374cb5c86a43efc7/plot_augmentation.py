# -*- coding: utf-8 -*-
"""
Spherical augmentations
=======================

Credit: A Grigis & C Ambroise

A simple example on how to use augmentations in the spherical domain.
"""

import numpy as np
import matplotlib.pyplot as plt
from surfify.utils import icosahedron, neighbors, min_depth_to_get_n_neighbors
from surfify.plotting import plot_trisurf
from surfify.augmentation import (
    SurfCutOut, SurfNoise, SurfBlur, SurfRotation, HemiMixUp, GroupMixUp,
    Transformer, interval)

vertices, triangles = icosahedron(order=3)
neighs = neighbors(vertices, triangles, direct_neighbor=True)
max_depth = min_depth_to_get_n_neighbors(np.ceil(len(vertices) / 4))


def display(vertices, triangles, texture, aug, add_axis=False, alpha=1, *args,
            **kwargs):
    """ Display augmented data.
    """
    fig, axs = plt.subplots(
        2, 2, subplot_kw={"projection": "3d", "aspect": "auto"},
        figsize=(10, 10))
    axs = axs.flatten()
    plot_trisurf(vertices, triangles, texture, ax=axs[0], fig=fig,
                 alpha=alpha, colorbar=False, edgecolors="white",
                 linewidths=0.2)
    if add_axis:
        axs[0].plot([0, 0], [0, 0], [-1, 1], c="red")
        axs[0].plot([0, 0], [-1, 1], [0, 0], c="blue")
        axs[0].plot([-1, 1], [0, 0], [0, 0], c="green")
    for idx in range(1, len(axs)):
        _texture = aug(texture, *args, **kwargs)
        if add_axis:
            axs[idx].plot([0, 0], [0, 0], [-1, 1], c="red")
            axs[idx].plot([0, 0], [-1, 1], [0, 0], c="blue")
            axs[idx].plot([-1, 1], [0, 0], [0, 0], c="green")
        plot_trisurf(vertices, triangles, _texture, ax=axs[idx], fig=fig,
                     alpha=alpha, colorbar=False, edgecolors="white",
                     linewidths=0.2)
    fig.tight_layout()


#############################################################################
# SurfCutOut
# ----------
#

texture = np.array([1, ] * len(vertices))
aug = SurfCutOut(vertices=vertices, triangles=triangles, neighs=neighs,
                 patch_size=interval((2, max_depth), int),
                 n_patches=interval((1, 3), int),
                 sigma=1)
display(vertices, triangles, texture, aug, alpha=0.3)


#############################################################################
# SurfNoise
# ---------
#

texture = np.random.uniform(0, 3, len(vertices))
aug = SurfNoise(sigma=interval((1, 3), float))
display(vertices, triangles, texture, aug)


#############################################################################
# SurfBlur
# -----------
#

texture = np.random.uniform(0, 2, len(vertices))
aug = SurfBlur(vertices=vertices, triangles=triangles,
               sigma=interval((0.1, 1), float))
display(vertices, triangles, texture, aug)


#############################################################################
# SurfRotation
# ------------
#

texture = np.array([1, ] * len(vertices))
aug = SurfRotation(vertices, triangles, phi=interval((5, 180), float), theta=0,
                   psi=0)
texture[neighs[0]] = 0
display(vertices, triangles, texture, aug, add_axis=True, alpha=0.5)


#############################################################################
# HemiMixUp
# ---------
#

texture = np.random.uniform(0, 3, len(vertices))
controlateral_texture = np.random.uniform(0, 3, len(vertices))
aug = HemiMixUp(prob=interval((0.2, 0.5), float), n_vertices=len(vertices))
display(vertices, triangles, texture, aug,
        controlateral_data=controlateral_texture)


#############################################################################
# GroupMixUp
# ----------
#

texture = np.random.uniform(0, 3, len(vertices))
group_textures = np.random.uniform(0, 3, (10, len(vertices)))
aug = GroupMixUp(prob=interval((0.2, 0.5), float), n_vertices=len(vertices))
display(vertices, triangles, texture, aug, group_data=group_textures,
        n_samples=1)


#############################################################################
# GroupMixUp group
# ----------------
#

textures = np.random.uniform(0, 3, (100, len(vertices)))
neigh_ind = GroupMixUp.groupby(textures, n_neighbors=4, n_components=30)
print(neigh_ind)


#############################################################################
# Transformer
# -----------
#

texture = np.random.uniform(0, 3, len(vertices))
aug1 = SurfCutOut(vertices=vertices, triangles=triangles, neighs=neighs,
                  patch_size=interval((1, max_depth), int),
                  n_patches=interval((1, 3), int))
aug2 = SurfNoise(sigma=interval((1, 3), float))
aug3 = SurfBlur(vertices=vertices, triangles=triangles,
                sigma=interval((0.1, 1), float))
trans = Transformer()
trans.register(aug1, probability=.75)
trans.register(aug2, probability=.75)
trans.register(aug3, probability=.5)
display(vertices, triangles, texture, trans)

plt.show()
