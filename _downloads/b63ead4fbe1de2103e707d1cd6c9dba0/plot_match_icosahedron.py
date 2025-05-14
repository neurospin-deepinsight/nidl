# -*- coding: utf-8 -*-
"""
Icosahedron matching
====================

Credit: C Ambroise

A simple example on how to match two icosahedrons of the same order.
"""
import os
import math
import warnings
import numpy as np
from scipy.spatial import transform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from surfify.plotting import plot_trisurf
from surfify.utils import icosahedron, ico2ico

#############################################################################
# We first build the reference icosahedron.

order = 3
vertices_norm, triangles_norm = icosahedron(order, standard_ico=True)
print(vertices_norm.shape, triangles_norm.shape)

#############################################################################
# Then we fetch freesurfer's icosahedron of the same order.

vertices, triangles = icosahedron(order)
print(vertices.shape, triangles.shape)

#############################################################################
# We try to find the optimal rotation between the two icosahedrons using
# the scipy module.

rotation, rmse = transform.Rotation.align_vectors(vertices_norm, vertices)
print(rmse)

#############################################################################
# Okay, does not seem to be working, because the rmse is supposed to be very
# close or equal to zero
#
# We print the vertices to try to find the issue here: it seems that the order
# of the vertices is not the same in the two matrices. That is why the previous
# algorithm did not work properly, since it can only match to the corresponding
# row in the other matrix.

print(vertices_norm)
print(vertices)

#############################################################################
# Here we plot the sufaces together to show that they have the same structure
# but their vertices are not at the same places.

fig, ax = plt.subplots(1, 1, subplot_kw={
        "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
plot_trisurf(vertices_norm, triangles_norm, fig=fig, ax=ax, alpha=0.3,
             edgecolors="blue")
plot_trisurf(vertices, triangles, fig=fig, ax=ax, alpha=0.3,
             edgecolors="green")

#############################################################################
# To compute the rotation between the two structures, we do not need all the
# vertices. So we consider a small subset of 4 points that have correspondances
# in both icosahedrons (for instance they have the same absolute
# values, only sign differs, for each dimension).
# The 4 firsts work for the reference icosahedron.

vertices_of_interest_norm = vertices_norm[:4]

#############################################################################
# Now we search for 4 similar vertices in the FreeSurfer icosahedron.

for i in range(len(vertices)):
    coords_of_interest = vertices[i]
    idx_of_interest = (np.abs(vertices) == np.abs(coords_of_interest)).all(1)
    if idx_of_interest.sum() == 4:
        vertices_of_interest = vertices[idx_of_interest]
        fs_row_idx = i
        break
print(fs_row_idx)

#############################################################################
# Now we need to find a rotation between these two set of points.
# Many can be possible, depending on the ordering of the points. To do this,
# we compute the optimal rotation matrix between our reference points and
# the others variously permuted, until we find a rotation that works.

import itertools
permutations = itertools.permutations(range(4))
n_permutations = math.factorial(4)
it = 0
best_rmse = rmse
best_rotation = rotation
while rmse > 0 and it < n_permutations:
    it += 1
    order = np.array(next(permutations))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        rotation, rmse = transform.Rotation.align_vectors(
                vertices_of_interest_norm, vertices_of_interest[order])
    if rmse < best_rmse:
        best_rmse = rmse
        best_rotation = rotation
print("Number of permutations tested {}/{}".format(it, math.factorial(4)))
print(best_rotation.as_matrix())
print(best_rmse)

#############################################################################
# Now we found a rotation that works, we can simply apply it to the icosahedron
# so it matches the reference one.

fig, ax = plt.subplots(1, 1, subplot_kw={
        "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
plot_trisurf(vertices_norm, triangles_norm, fig=fig, ax=ax, alpha=0.3,
             edgecolors="blue")
plot_trisurf(best_rotation.apply(vertices), triangles, fig=fig, ax=ax,
             alpha=0.3, edgecolors="green")

#############################################################################
# To easily solve the issue outlined in this example, you can find a
# function in the `surfify.utils` module. `ico2ico` allows you to find a
# proper rotation between a reference icosahedron and another one.
# We plot only half of the triangles of each icosahedron so it clearly appears
# that they are the same.
rotation = ico2ico(vertices, vertices_norm)
fig, ax = plt.subplots(1, 1, subplot_kw={
        "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
plot_trisurf(vertices_norm, triangles_norm[::2], fig=fig, ax=ax, alpha=0.3,
             edgecolors="blue")
plot_trisurf(rotation.apply(vertices), triangles[::2], fig=fig, ax=ax,
             alpha=0.3, edgecolors="green")
plt.show()
