"""
Spherical neighbors
===================


This example shows how to build and visualize spherical neighbors on an
icosahedral mesh, focusing on triangular patches and direct neighbors (DiNe).

Build an icosahedral sphere
---------------------------

We start from creating an icosahedron of a specific order. Higher order means
more vertices and triangles on the sphere.
"""

from nidl.surface.base import Ico

kwargs = {
    "order": 3,
    "standard_ico": True,
}

verts, tris = Ico.mesh(**kwargs)
print(f"Mesh: {verts.shape} - {tris.shape}")

# %%
# This gives us a spherical mesh: a discretized sphere built from an
# icosahedron.
#
# Extract spherical patches
# -------------------------
#
# A patch is a small triangular local neighborhood around each vertex,
# defined by the number of triangular rings.

patches_1 = Ico.patches(size=1, **kwargs)
print(f"One ring patches: {patches_1.shape}")
patches_2 = Ico.patches(size=2, **kwargs)
print(f"Two rings patches: {patches_2.shape}")

# %%
# This is useful for transformer definition. Note that the patch vertices can
# by sorted by angular order.
#
# Build direct neighbors (DiNe)
# -----------------------------
#
# Now we focus on direct neighbors around each vertex, containing vertices that
# are directly connected to a given vertex by an edge, and defined by the
# number of rings. Here, neighbor vertices are sorted by angular order.

neighs_1 = Ico.neighbors(depth=1, direct_neighbor=True, **kwargs)
print(f"One ring DiNEs: {neighs_1.shape}")
neighs_2 = Ico.neighbors(depth=2, direct_neighbor=True, **kwargs)
print(f"Two rings DiNEs: {neighs_2.shape}")

# %%
# The obtained array contains for each vertex his direct neighborhood. This
# is useful for spherical CNN definition.
#
# Visualize the mesh and neighbors
# --------------------------------
#
# We visualize the generated topological information: spherical mesh,
# patches (in red) and DiNes (in green).

from nidl.surface.plotting import IcoRenderer

ren = IcoRenderer(
    figsize=(10, 10),
)
ren.add_trisurf(
    verts,
    tris,
)
ren.add_points(
    [verts[idx] for idx in patches_1[0]],
    size=100,
    color="red",
)
ren.add_points(
    [verts[idx] for idx in patches_2[25]],
    size=100,
    color="red",
)
ren.add_points(
     [verts[idx] for idx in neighs_1[4]],
    labels=range(7),
    size=100,
    color="green",
)
ren.add_points(
     [verts[idx] for idx in neighs_1[40]],
    labels=range(7),
    size=100,
    color="green",
)
ren.add_points(
     [verts[idx] for idx in neighs_2[50]],
    labels=range(19),
    size=100,
    color="green",
)
ren.show()
