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
    procrustes_similarity,
    procrustes_r2,
    kruskal_similarity
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

def random_orthogonal_matrix(n):
    A = np.random.randn(n, n)
    
    Q, R = np.linalg.qr(A)
    
    d = np.diag(R)
    ph = d / np.abs(d)
    Q *= ph
    
    return Q

class TestProcrustesSimilarity(unittest.TestCase):
    def dim_type_mismatch_raises(self):
        z1 = np.random.uniform((4, 3, 2))
        z2 = np.random.uniform((4, 3))

        with self.assertRaises(ValueError):
            procrustes_similarity(z1, z2)
        with self.assertRaises(ValueError):
            procrustes_similarity(torch.Tensor(z1), z2)
        with self.assertRaises(ValueError):
            procrustes_similarity(torch.Tensor(z1), torch.Tensor(z2))

    def length_mismatch_raises(self):
        z1 = np.random.uniform((10, 3))
        z2 = np.random.uniform((9,3))

        with self.assertRaises(ValueError):
            procrustes_similarity(z1, z2)
        with self.assertRaises(ValueError):
            procrustes_similarity(torch.Tensor(z1), z2)
        with self.assertRaises(ValueError):
            procrustes_similarity(torch.Tensor(z1), torch.Tensor(z2))

    def test_type_consistency(self):
        # for np.ndarrays, check that dtype and return type are preserved
        for dtype in (np.float32, np.float64):
            X = np.random.uniform(size=(100, 4)).astype(dtype)
            Y = np.random.uniform(size=(100, 4)).astype(dtype)
            self.assertTrue(isinstance(procrustes_similarity(X, Y), np.ndarray))
            self.assertTrue(procrustes_similarity(X, Y).dtype == dtype)

        # for torch.Tensors, check that dtype, return type and device are preserved
        for dtype in (torch.float32, torch.float64):
            devices = ("cuda", "cpu") if torch.cuda.is_available() else ("cpu",)
            for device in devices:
                X = torch.rand((100, 4), device=device, dtype=dtype)
                Y = torch.rand((100, 4), device=device, dtype=dtype)
                self.assertTrue(isinstance(procrustes_similarity(X, Y), torch.Tensor))
                self.assertTrue(procrustes_similarity(X, Y).dtype == dtype)
                self.assertTrue(procrustes_similarity(X, Y).device.type == device)

        # check that calculation is equal regardless of dtype, return type and device
        X = np.random.uniform(size=(100, 4)).astype("float64")
        Y = np.random.uniform(size=(100, 4)).astype("float64")

        sim = procrustes_similarity(X,Y)
        others = [
            procrustes_similarity(X.astype("float32"), Y),
            procrustes_similarity(X, Y.astype("float32")),
            procrustes_similarity(torch.Tensor(X), Y),
            procrustes_similarity(X, torch.Tensor(Y).to(dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")),
            procrustes_similarity(torch.Tensor(X).to(dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu"), torch.Tensor(Y).to(dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")),
            procrustes_similarity(torch.Tensor(X), torch.Tensor(Y)),
        ]

        for other in others:
            if isinstance(other, torch.Tensor):
                other = other.to("cpu")
            self.assertAlmostEqual(float(sim), float(other), places=4)

    def test_mathematical_properties(self):
        # check that sim == 1 if Y = sQX + b with Q orthogonal

        Q = random_orthogonal_matrix(4)
        X = np.random.uniform(size=(100, 4))
        
        for s in (2, 3, 4):
            for b in [np.random.uniform(size=(4,)) for _ in range(3)]:
                Y = s * X@Q + b
                self.assertAlmostEqual(float(procrustes_similarity(X, Y)), 1.0, places=7)

        # check that sim != 1 if Y = sQX + b with Q *not* orthogonal
        for _ in range(3):
            while True:
                Q = np.random.randn(4, 4)
                if np.allclose(Q @ Q.T, np.eye(4)):
                    continue
                for s in (2, 3, 4):
                    for b in [np.random.uniform(size=(4,)) for _ in range(3)]:
                        Y = s * X@Q + b
                        self.assertNotAlmostEqual(float(procrustes_similarity(X, Y)), 1.0, places=7)
                break

        # check invariance to padding
        Y = np.random.uniform(size=(100, 4))
        sims = [procrustes_similarity(np.hstack([X, np.zeros((100, i))]), np.hstack([Y, np.zeros((100, j))])) for i in range(3) for j in range(3)]
        for sim in sims[1:]:
            self.assertAlmostEqual(float(sims[0]), float(sim), places=7)

class TestProcrustesR2(unittest.TestCase):
    def dim_type_mismatch_raises(self):
        z1 = np.random.uniform((4, 3, 2))
        z2 = np.random.uniform((4, 3))
        
        with self.assertRaises(ValueError):
            procrustes_r2(z1, z2)
        with self.assertRaises(ValueError):
            procrustes_r2(torch.Tensor(z1), z2)
        with self.assertRaises(ValueError):
            procrustes_r2(torch.Tensor(z1), torch.Tensor(z2))

    def length_mismatch_raises(self):
        z1 = np.random.uniform((10, 3))
        z2 = np.random.uniform((9,3))

        with self.assertRaises(ValueError):
            procrustes_r2(z1, z2)
        with self.assertRaises(ValueError):
            procrustes_r2(torch.Tensor(z1), z2)
        with self.assertRaises(ValueError):
            procrustes_r2(torch.Tensor(z1), torch.Tensor(z2))

    def test_type_consistency(self):
        # for np.ndarrays, check that dtype and return type are preserved
        for dtype in (np.float32, np.float64):
            X = np.random.uniform(size=(100, 4)).astype(dtype)
            Y = np.random.uniform(size=(100, 4)).astype(dtype)
            self.assertTrue(isinstance(procrustes_r2(X, Y), np.ndarray))
            self.assertTrue(procrustes_r2(X, Y).dtype == dtype)

        # for torch.Tensors, check that dtype, return type and device are preserved
        for dtype in (torch.float32, torch.float64):
            devices = ("cuda", "cpu") if torch.cuda.is_available() else ("cpu",)
            for device in devices:
                X = torch.rand((100, 4), device=device, dtype=dtype)
                Y = torch.rand((100, 4), device=device, dtype=dtype)
                self.assertTrue(isinstance(procrustes_r2(X, Y), torch.Tensor))
                self.assertTrue(procrustes_r2(X, Y).dtype == dtype)
                self.assertTrue(procrustes_r2(X, Y).device.type == device)

        # check that calculation is equal in regardless of dtype, return type and device
        X = np.random.uniform(size=(100, 4)).astype("float64")
        Y = np.random.uniform(size=(100, 4)).astype("float64")

        sim = procrustes_r2(X,Y)
        others = [
            procrustes_r2(X.astype("float32"), Y),
            procrustes_r2(X, Y.astype("float32")),
            procrustes_r2(torch.Tensor(X), Y),
            procrustes_r2(X, torch.Tensor(Y).to(dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")),
            procrustes_r2(torch.Tensor(X).to(dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu"), torch.Tensor(Y).to(dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")),
            procrustes_r2(torch.Tensor(X), torch.Tensor(Y)),
        ]

        for other in others:
            if isinstance(other, torch.Tensor):
                other = other.to("cpu")
            self.assertAlmostEqual(float(sim), float(other), places=4)

    def test_mathematical_properties(self):
        # check that sim == 1 if Y = QX + b with Q orthogonal

        Q = random_orthogonal_matrix(4)
        X = np.random.uniform(size=(100, 4))
        
        for b in [np.random.uniform(size=(4,)) for _ in range(3)]:
            Y = X@Q + b
            self.assertAlmostEqual(float(procrustes_similarity(X, Y)), 1.0, places=7)

        # check that sim != 1 if Y = QX + b with Q *not* orthogonal
        for _ in range(3):
            while True:
                Q = np.random.randn(4, 4)
                if np.allclose(Q @ Q.T, np.eye(4)):
                    continue

                for b in [np.random.uniform(size=(4,)) for _ in range(3)]:
                    Y = X@Q + b
                    self.assertNotAlmostEqual(float(procrustes_similarity(X, Y)), 1.0, places=7)
                break

        # check invariance to padding
        Y = np.random.uniform(size=(100, 4))
        sims = [procrustes_similarity(np.hstack([X, np.zeros((100, i))]), np.hstack([Y, np.zeros((100, j))])) for i in range(3) for j in range(3)]
        for sim in sims[1:]:
            self.assertAlmostEqual(float(sims[0]), float(sim), places=7)

class TestKruskalSimilarity(unittest.TestCase):
    def dim_type_mismatch_raises(self):
        z1 = np.random.uniform((4, 3, 2))
        z2 = np.random.uniform((4, 3))

        for spherical in (True, False):
            with self.assertRaises(ValueError):
                kruskal_similarity(z1, z2, spherical=spherical)
            with self.assertRaises(ValueError):
                kruskal_similarity(torch.Tensor(z1), z2, spherical=spherical)
            with self.assertRaises(ValueError):
                kruskal_similarity(torch.Tensor(z1), torch.Tensor(z2), spherical=spherical)

    def length_mismatch_raises(self):
        z1 = np.random.uniform((10, 3))
        z2 = np.random.uniform((9,3))

        for spherical in (True, False):
            with self.assertRaises(ValueError):
                kruskal_similarity(z1, z2, spherical=spherical)
            with self.assertRaises(ValueError):
                kruskal_similarity(torch.Tensor(z1), z2, spherical=spherical)
            with self.assertRaises(ValueError):
                kruskal_similarity(torch.Tensor(z1), torch.Tensor(z2), spherical=spherical)

    def test_type_consistency(self):
        for spherical in (True, False):
            # for np.ndarrays, check that dtype and return type are preserved
            for dtype in (np.float32, np.float64):
                X = np.random.uniform(size=(100, 4)).astype(dtype)
                Y = np.random.uniform(size=(100, 4)).astype(dtype)
                self.assertTrue(isinstance(kruskal_similarity(X, Y, spherical=spherical), np.ndarray))
                self.assertTrue(kruskal_similarity(X, Y, spherical=spherical).dtype == dtype)

            # for torch.Tensors, check that dtype, return type and device are preserved
            for dtype in (torch.float32, torch.float64):
                devices = ("cuda", "cpu") if torch.cuda.is_available() else ("cpu",)
                for device in devices:
                    X = torch.rand((100, 4), device=device, dtype=dtype)
                    Y = torch.rand((100, 4), device=device, dtype=dtype)
                    self.assertTrue(isinstance(kruskal_similarity(X, Y, spherical=spherical), torch.Tensor))
                    self.assertTrue(kruskal_similarity(X, Y, spherical=spherical).dtype == dtype)
                    self.assertTrue(kruskal_similarity(X, Y, spherical=spherical).device.type == device)

            # check that calculation is equal regardless of dtype, return type and device
            X = np.random.uniform(size=(100, 4)).astype("float64")
            Y = np.random.uniform(size=(100, 4)).astype("float64")

            sim = kruskal_similarity(X,Y, spherical=spherical)
            others = [
                kruskal_similarity(X.astype("float32"), Y, spherical=spherical),
                kruskal_similarity(X, Y.astype("float32"), spherical=spherical),
                kruskal_similarity(torch.Tensor(X), Y, spherical=spherical),
                kruskal_similarity(X, torch.Tensor(Y).to(dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu"), spherical=spherical),
                kruskal_similarity(
                    torch.Tensor(X).to(dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu"),
                    torch.Tensor(Y).to(dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu"),
                    spherical=spherical
                    ),
                kruskal_similarity(torch.Tensor(X), torch.Tensor(Y), spherical=spherical),
            ]

            for other in others:
                if isinstance(other, torch.Tensor):
                    other = other.to("cpu")
                self.assertAlmostEqual(float(sim), float(other), places=4)

    def test_mathematical_properties(self):
        # check that kruskal_similarity(X, X@Q+b, spherical=False) == 1

        Q = random_orthogonal_matrix(4)
        X = np.random.uniform(size=(100, 4))
        
        for b in [np.random.uniform(size=(4,)) for _ in range(3)]:
            Y = X@Q + b
            self.assertAlmostEqual(float(kruskal_similarity(X, Y, spherical=False)), 1.0, places=7)
        
        # check that kruskal_similarity(X, X@Q, spherical=True) == 1
        self.assertAlmostEqual(float(kruskal_similarity(X, X@Q, spherical=True)), 1.0, places=7)

        # check that sim != 1 if Y = QX + b with Q *not* orthogonal
        for _ in range(3):
            while True:
                Q = np.random.randn(4, 4)
                if np.allclose(Q @ Q.T, np.eye(4)):
                    continue

                for b in [np.random.uniform(size=(4,)) for _ in range(3)]:
                    Y = X@Q + b
                    self.assertNotAlmostEqual(float(kruskal_similarity(X, Y, spherical=True)), 1.0, places=7)
                    self.assertNotAlmostEqual(float(kruskal_similarity(X, Y, spherical=False)), 1.0, places=7)
                break

        Y = np.random.uniform(size=(100, 4))

        # check invariance to padding
        for spherical in (True, False):
            sims = [kruskal_similarity(np.hstack([X, np.zeros((100, i))]), np.hstack([Y, np.zeros((100, j))]), spherical=spherical) for i in range(3) for j in range(3)]
            for sim in sims[1:]:
                self.assertAlmostEqual(float(sims[0]), float(sim), places=7)

        # check invariance to rescaling in the spherical case
        sims = [kruskal_similarity(X, np.diag(np.random.uniform(size=(100,), low=.1, high=5)) @ Y, spherical=True) for i in range(5)]

        for sim in sims[1:]:
            self.assertAlmostEqual(float(sims[0]), float(sim), places=7)

if __name__ == "__main__":
    unittest.main()
