##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import unittest
import numpy as np
import torch

from nidl.metrics import (  # <-- change this to your actual module path
    alignment_score,
    uniformity_score,
    contrastive_accuracy_score,
)


class TestAlignmentScore(unittest.TestCase):
    def test_torch_perfect_alignment_no_normalize(self):
        # z1 == z2 -> distance zero -> score == 0
        z1 = torch.randn(8, 5)
        z2 = z1.clone()
        score = alignment_score(z1, z2, normalize=False)
        self.assertIsInstance(score, torch.Tensor)
        self.assertEqual(score.dim(), 0)
        self.assertAlmostEqual(score.item(), 0.0, places=7)

    def test_torch_perfect_alignment_normalize(self):
        # even with normalization, identical tensors stay identical -> score == 0
        z1 = torch.randn(8, 5)
        z2 = z1.clone()
        score = alignment_score(z1, z2, normalize=True)
        self.assertAlmostEqual(score.item(), 0.0, places=7)

    def test_numpy_matches_manual_computation_alpha2(self):
        # Small, hand-checkable example
        z1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        z2 = np.array([[0.0, 1.0], [1.0, 0.0]])
        # No normalization so we can compute analytically
        score = alignment_score(z1, z2, normalize=False, alpha=2)

        diff = z1 - z2
        dist = np.linalg.norm(diff, axis=1) ** 2
        expected = dist.mean()
        self.assertIsInstance(score, np.generic)
        self.assertAlmostEqual(float(score), float(expected), places=7)

    def test_numpy_alpha1_vs_alpha2_relation(self):
        # With alpha=1 vs alpha=2 we should get different scores but consistent ordering
        z1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        z2 = np.array([[0.5, 0.5], [0.5, 0.5]])
        s1 = alignment_score(z1, z2, normalize=False, alpha=1)
        s2 = alignment_score(z1, z2, normalize=False, alpha=2)
        self.assertNotAlmostEqual(float(s1), float(s2))
        # Both must be >= 0
        self.assertGreaterEqual(float(s1), 0.0)
        self.assertGreaterEqual(float(s2), 0.0)

    def test_type_mismatch_raises(self):
        z1 = torch.randn(4, 3)
        z2 = np.random.randn(4, 3)
        with self.assertRaises(TypeError):
            alignment_score(z1, z2)


class TestUniformityScore(unittest.TestCase):
    def _manual_uniformity_numpy(self, z, t=2.0, eps=1e-12):
        # Manual computation following the docstring: normalize, then log E exp(-t ||zi - zj||^2)
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + eps)
        sim = z @ z.T
        dist_sq = 2 - 2 * sim
        mask = ~np.eye(len(z), dtype=bool)
        dist_sq = dist_sq[mask]
        return np.log(np.exp(-t * dist_sq).mean())

    def test_numpy_matches_manual(self):
        z = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        score = uniformity_score(z, normalize=True, t=2.0)
        expected = self._manual_uniformity_numpy(z, t=2.0)
        self.assertIsInstance(score, np.generic)
        self.assertAlmostEqual(float(score), float(expected), places=7)

    def test_torch_and_numpy_close(self):
        # Same data in torch vs numpy should give almost the same value
        z_np = np.random.randn(10, 4).astype(np.float32)
        z_t = torch.from_numpy(z_np.copy())
        s_np = uniformity_score(z_np, normalize=True, t=2.0)
        s_t = uniformity_score(z_t, normalize=True, t=2.0)
        self.assertIsInstance(s_t, torch.Tensor)
        self.assertEqual(s_t.dim(), 0)
        self.assertAlmostEqual(float(s_np), float(s_t.item()), places=6)

    def test_more_uniform_has_lower_score(self):
        # Very clustered vs more spread: more spread should have lower uniformity score
        clustered = np.array([[1.0, 0.0]] * 4)
        spread = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]
        )
        s_clustered = uniformity_score(clustered, normalize=True)
        s_spread = uniformity_score(spread, normalize=True)
        self.assertLess(float(s_spread), float(s_clustered))

    def test_type_mismatch_raises(self):
        with self.assertRaises(TypeError):
            uniformity_score("not an array")  # type: ignore[arg-type]


class TestContrastiveAccuracyScore(unittest.TestCase):
    def test_torch_perfect_alignment_top1(self):
        # z1 == z2, unique pairs: accuracy must be 1.0 for topk = 1
        z1 = torch.randn(5, 3)
        z2 = z1.clone()
        score = contrastive_accuracy_score(z1, z2, normalize=True, topk=1)
        self.assertIsInstance(score, torch.Tensor)
        self.assertEqual(score.dim(), 0)
        self.assertAlmostEqual(score.item(), 1.0, places=7)

    def test_numpy_perfect_alignment_topk_greater_than_n(self):
        # topk > n_samples should be clipped → still accuracy 1.0
        z1 = np.random.randn(4, 2).astype(np.float32)
        z2 = z1.copy()
        score = contrastive_accuracy_score(z1, z2, normalize=True, topk=10)
        self.assertIsInstance(score, np.generic)
        self.assertAlmostEqual(float(score), 1.0, places=7)

    def test_nontrivial_permutation_accuracy_between_0_and_1(self):
        # Construct embeddings where some but not all matches are nearest neighbors
        # z1 and z2 are permutations of each other
        z1 = np.array(
            [
                [1.0, 0.0],   # A
                [0.0, 1.0],   # B
                [1.0, 1.0],   # C
            ],
            dtype=np.float32,
        )
        # Swap B and C
        z2 = np.array(
            [
                [1.0, 0.0],   # A'
                [1.0, 1.0],   # C'
                [0.0, 1.0],   # B'
            ],
            dtype=np.float32,
        )
        score = contrastive_accuracy_score(z1, z2, normalize=True, topk=1)
        self.assertGreater(float(score), 0.0)
        self.assertLess(float(score), 1.0)

    def test_torch_vs_numpy_close(self):
        z_np = np.random.randn(6, 4).astype(np.float32)
        z_t = torch.from_numpy(z_np.copy())
        s_np = contrastive_accuracy_score(z_np, z_np.copy(), normalize=True, topk=2)
        s_t = contrastive_accuracy_score(z_t, z_t.clone(), normalize=True, topk=2)
        self.assertAlmostEqual(float(s_np), float(s_t.item()), places=6)

    def test_topk_less_than_one_raises(self):
        z = np.random.randn(3, 2).astype(np.float32)
        with self.assertRaises(ValueError):
            contrastive_accuracy_score(z, z, topk=0)

    def test_shape_mismatch_raises(self):
        z1 = np.random.randn(3, 2).astype(np.float32)
        z2 = np.random.randn(4, 2).astype(np.float32)
        with self.assertRaises(ValueError):
            contrastive_accuracy_score(z1, z2)

    def test_ndim_not_2_raises(self):
        z1 = np.random.randn(3, 2, 2).astype(np.float32)
        z2 = np.random.randn(3, 2, 2).astype(np.float32)
        with self.assertRaises(ValueError):
            contrastive_accuracy_score(z1, z2)

    def test_empty_embeddings_raises(self):
        z1 = np.zeros((0, 4), dtype=np.float32)
        z2 = np.zeros((0, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            contrastive_accuracy_score(z1, z2)

    def test_type_mismatch_raises(self):
        z1 = torch.randn(4, 3)
        z2 = np.random.randn(4, 3).astype(np.float32)
        with self.assertRaises(TypeError):
            contrastive_accuracy_score(z1, z2)


if __name__ == "__main__":
    unittest.main()
