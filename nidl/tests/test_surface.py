##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import os
import numpy as np
import unittest
from nidl.surface import Ico
from nidl.surface.transforms import (
    RandomCutOut,
    RandomRotation,
)


class TestIco(unittest.TestCase):

    def tearDown(self):
        Ico.clear()

    def test_caching(self):
        vertices, triangles = Ico.mesh(order=3)
        key = "Ico.mesh(order=3, standard_ico=False)"
        self.assertTrue(key in Ico._cache)
        Ico._cache[key] = (None, None)
        vertices, triangles = Ico.mesh(order=3)
        self.assertTrue(vertices is None and triangles is None)

    def test_mesh(self):
        for order in range(5):
            vertices, _ = Ico.mesh(order)
            self.assertTrue(len(vertices) == Ico.n_vertices(order))
            self.assertTrue(order == Ico.order(len(vertices)))

    def test_neighbors(self):
        expected_results = {
            1: {"n_neighbors": 7},
            2: {"n_neighbors": 19},
            3: {"n_neighbors": 37},        
        }
        for depth in expected_results:
            neighs = Ico.neighbors(
                order=4,
                depth=depth,
                direct_neighbor=True,
                n_jobs=2,
            )
            res = expected_results[depth]
            expected_shape = (2562, res["n_neighbors"])
            self.assertTrue(expected_shape == neighs.shape)

    def test_patches(self):
        expected_results = {
            3: {"n_patches": 1280, "n_vertices": 45},
            4: {"n_patches": 320, "n_vertices": 153},
            5: {"n_patches": 80, "n_vertices": 561},
            6: {"n_patches": 20, "n_vertices": 2145},
        }
        for size in expected_results:
            patches = Ico.patches(
                order=6,
                size=size,
                direct_neighbor=True,
                n_jobs=2,
            )
            res = expected_results[size]
            expected_shape = (res["n_patches"], res["n_vertices"])
            self.assertTrue(expected_shape == patches.shape)


class TestTransforms(unittest.TestCase):

    def test_cutout(self):
        vertices, triangles = Ico.mesh(order=3)
        x = np.ones((len(vertices), ), dtype=int)
        transform = RandomCutOut(cuts=1, size=3)
        x_cutout = transform(x)
        self.assertTrue((x == x_cutout).sum() < len(vertices))

    def test_rotation(self):
        vertices, triangles = Ico.mesh(order=3)
        x = np.ones((len(vertices), ), dtype=int)
        transform = RandomRotation(phi=360, theta=0, psi=0)
        x_rotated = transform(x)
        self.assertTrue(np.allclose(x, x_rotated))


if __name__ == "__main__":
    unittest.main(verbosity=2)
