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
from torch.distributions import Normal, Laplace, Bernoulli

from nidl.losses import (
    BarlowTwinsLoss,
    DINOLoss,
    InfoNCE,
    YAwareInfoNCE,
    KernelMetric,
    BetaVAELoss
)

class TestSSLLosses(unittest.TestCase):
    """ Test backbones.
    """

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_infonce(self):
        """ Test InfoNCE loss is computed correctly.
        """
        for temperature in [0.1, 1.0, 5.0]:
            for batch_size in [1, 10]:
                for n_embedding in [1, 10]:
                    z1 = torch.rand(
                        batch_size, n_embedding)
                    z2 = torch.rand(
                        batch_size, n_embedding)
                    infonce = InfoNCE(temperature=temperature)
                    # Perfect alignment
                    loss_low = infonce(z1, z1)
                    # random alignment
                    loss_high = infonce(z1, z2)
                    assert loss_low <= loss_high, (
                        f"InfoNCE loss should be lower for aligned embeddings, "
                        f"got {loss_low} vs {loss_high}"
                    )
                    assert loss_low >= 0, "InfoNCE loss should be positive."
    
    def test_barlowtwins(self):
        """Test BarlowTwins loss is computed correctly.
        """
        lambd = 0.
        for batch_size in [5, 10]:
            for n_embedding in [5, 10]:
                z1 = torch.rand(
                    batch_size, n_embedding)
                barlowtwins = BarlowTwinsLoss(lambd)
                loss = barlowtwins(z1, z1)
                assert np.allclose(loss.numpy(), 0., atol=1e-10), (
                    "For an autocorrelation, diagonal elements should be equal "
                    "to 1, thus the invariance term should be equal to 0"
                )

    def test_barlowtwins_str(self):
        loss_fn = BarlowTwinsLoss(lambd=0.01)
        self.assertEqual(str(loss_fn), "BarlowTwinsLoss(lambd=0.01)")

    def test_barlowtwins_loss_batch_gt_1(self):
        # Typical batch size > 1
        torch.manual_seed(0)
        z1 = torch.randn(4, 5)
        z2 = torch.randn(4, 5)

        loss_fn = BarlowTwinsLoss(lambd=0.01)
        loss = loss_fn(z1, z2)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # should be scalar tensor
        self.assertGreater(loss.item(), 0)  # loss should be positive

    def test_barlowtwins_loss_batch_eq_1(self):
        # Edge case: batch size = 1
        torch.manual_seed(0)
        z1 = torch.randn(1, 5)
        z2 = torch.randn(1, 5)

        loss_fn = BarlowTwinsLoss(lambd=0.01)
        loss = loss_fn(z1, z2)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # should be scalar tensor

    def test_barlowtwins_gradients(self):
        # Check that the loss is differentiable
        z1 = torch.randn(4, 5, requires_grad=True)
        z2 = torch.randn(4, 5, requires_grad=True)

        loss_fn = BarlowTwinsLoss()
        loss = loss_fn(z1, z2)
        loss.backward()

        self.assertIsNotNone(z1.grad)
        self.assertIsNotNone(z2.grad)
        self.assertEqual(z1.grad.shape, z1.shape)
        self.assertEqual(z2.grad.shape, z2.shape)

    def test_yaware(self):
        """ Test y-Aware loss is computed correctly.
        """
        for temperature in [0.1, 1.0, 5.0]:
            for batch_size in [1, 10]:
                for n_embedding in [1, 10]:
                    for bandwidth in [0.1, 1.0]:
                        z1 = torch.rand(batch_size, n_embedding)
                        z2 = torch.rand(batch_size, n_embedding)
                        # Ensures all labels are different and sufficiently spaced
                        # compared to the bandwidth
                        labels = torch.arange(0, 3*batch_size, step=3).reshape(-1, 1)
                        yaware_infonce = YAwareInfoNCE(bandwidth=bandwidth, temperature=temperature)
                        # Perfect alignment for same labels
                        loss_low = yaware_infonce(z1, z1, labels=labels)
                        # random alignment
                        loss_high = yaware_infonce(z1, z2, labels=labels)
                        assert loss_low <= loss_high, (
                            f"y-Aware InfoNCE loss should be lower for aligned embeddings, "
                            f"got {loss_low} vs {loss_high}"
                        )
                        assert loss_low >= 0, "y-Aware InfoNCE loss should be positive."
        # Test bandwidth computation
        z1 = torch.rand(10, 2)
        z2 = torch.rand(10, 2)
        labels = torch.rand(10, 3)
        covar = (labels.T @ labels).numpy()
        for bandwidth in ["scott", "silverman", covar]:
            kernel = KernelMetric(bandwidth=bandwidth)
            loss = YAwareInfoNCE(bandwidth=kernel)
            with self.assertRaises(TypeError): # kernel not fitted
                loss(z1, z2, labels)
            kernel.fit(labels)
            kernel_loss = loss(z1, z2, labels)
            assert  kernel_loss >= 0, "y-Aware InfoNCE loss should be positive."
            if not isinstance(bandwidth, str): # SDP matrix as bandwidth
                loss = YAwareInfoNCE(bandwidth=bandwidth)
                assert loss(z1, z2, labels) == kernel_loss
                assert np.allclose(kernel.inv_sqr_bandwidth_ @ \
                                   kernel.inv_sqr_bandwidth_ @ kernel.bandwidth, np.eye(3), atol=1e-6)
                assert np.allclose(kernel.inv_sqr_bandwidth_ @ kernel.sqr_bandwidth_, np.eye(3), atol=1e-6)
                assert np.allclose(kernel.sqr_bandwidth_ @ kernel.sqr_bandwidth_, kernel.bandwidth, atol=1e-6)
        with self.assertRaises(ValueError):
            # no SDP bandwidth
            loss = YAwareInfoNCE(bandwidth=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            loss(z1, z2, labels)
        with self.assertRaises(ValueError):
            # negative values
            loss = YAwareInfoNCE(bandwidth=-np.eye(3))
            loss(z1, z2, labels)

    def test_eq_yaware_infonce(self):
        """ Test that YAwareInfoNCE is equal to InfoNCE when no labels are provided.
        """
        z1 = torch.rand(10, 5)
        z2 = torch.rand(10, 5)
        ya_infonce = YAwareInfoNCE()
        infonce = InfoNCE()
        loss_ya = ya_infonce(z1, z2)
        loss_inf = infonce(z1, z2)
        assert torch.allclose(loss_ya, loss_inf), (
            "YAwareInfoNCE should be equal to InfoNCE when no labels are provided, got "
            f"{loss_ya} vs {loss_inf}"
        )


class TestDINOLoss(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def _make_loss(
        self,
        output_dim=4,
        warmup_teacher_temp=0.04,
        teacher_temp=0.07,
        warmup_teacher_temp_epochs=3,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        return DINOLoss(
            output_dim=output_dim,
            warmup_teacher_temp=warmup_teacher_temp,
            teacher_temp=teacher_temp,
            warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
            student_temp=student_temp,
            center_momentum=center_momentum,
        )

    def test_temperature_warmup_schedule_and_post_warmup(self):
        warmup_epochs = 3
        warmup_start = 0.04
        warmup_end = 0.07
        loss_fn = self._make_loss(
            warmup_teacher_temp=warmup_start,
            teacher_temp=warmup_end,
            warmup_teacher_temp_epochs=warmup_epochs,
        )
        loss_fn2 = self._make_loss(
            teacher_temp=loss_fn.teacher_temp_schedule[0],
            student_temp=loss_fn.student_temp,
        )
        loss_fn2.center = loss_fn.center.clone()

        # Sanity-check schedule endpoints
        self.assertEqual(len(loss_fn.teacher_temp_schedule), warmup_epochs)
        self.assertAlmostEqual(float(loss_fn.teacher_temp_schedule[0]), warmup_start, places=7)
        self.assertAlmostEqual(float(loss_fn.teacher_temp_schedule[-1]), warmup_end, places=7)

        teacher_out = torch.randn(2, 2, 4)
        student_out = torch.randn(3, 2, 4)

        # Sanity-check warmup scheduler
        loss0 = loss_fn(teacher_out, student_out, epoch=0)
        loss1 = loss_fn2(teacher_out, student_out, epoch=0)
        loss_fn2.center = loss_fn.center.clone()
        loss2 = loss_fn2(teacher_out, student_out, epoch=3)
        loss_fn2.center = loss_fn.center.clone()
        loss3 = loss_fn2(teacher_out, student_out, epoch=4)
        self.assertTrue(torch.allclose(loss0, loss1, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(loss2, loss3, atol=1e-6, rtol=1e-6))
        self.assertFalse(torch.allclose(loss0, loss2, atol=1e-6, rtol=1e-6))


    def test_diagonal_is_ignored_and_normalization_matches(self):
        loss_fn = self._make_loss(output_dim=4, warmup_teacher_temp_epochs=3)
        loss_fn.center.zero_()

        # Random examples with non-symmetric structures
        teacher_out = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4]],
                [[0.0, 1.0, 0.0, 0.0], [0.4, 0.3, 0.2, 0.1]],
            ]
        )  # (t=2, b=2, d=4)

        student_out = torch.tensor(
            [
                [[0.3, 0.2, 0.1, 0.0], [0.0, 0.1, 0.2, 0.3]],
                [[0.0, 0.1, 0.0, 0.9], [0.9, 0.0, 0.1, 0.0]],
                [[0.2, 0.2, 0.2, 0.2], [0.4, 0.3, 0.2, 0.1]],
            ]
        )  # (s=3, b=2, d=4)

        student_out_perm = student_out[[1, 0, 2], :, :]
        teacher_out_perm = teacher_out[[1, 0], :, :]
        student_out_wrong_perm = student_out[[0, 2, 1], :, :]
        teacher_out_wrong_perm = teacher_out[[1, 0], :, :]
        loss = loss_fn(teacher_out, student_out, epoch=10)
        loss_fn.center.zero_()
        expected = loss_fn(teacher_out_perm, student_out_perm, epoch=10)
        loss_fn.center.zero_()
        unexpected = loss_fn(teacher_out_wrong_perm, student_out_wrong_perm, epoch=10)

        self.assertTrue(torch.allclose(loss, expected, atol=1e-6, rtol=1e-6))
        self.assertFalse(torch.allclose(loss, unexpected, atol=1e-6, rtol=1e-6))


    def test_center_updates_with_momentum_non_distributed(self):
        momentum = 0.9
        loss_fn = self._make_loss(output_dim=4, center_momentum=momentum)
        self.assertTrue(torch.allclose(loss_fn.center, torch.zeros_like(loss_fn.center)))

        teacher_out = torch.randn(2, 3, 4)  # (n_views, batch, dim)
        batch_center = teacher_out.mean(dim=(0, 1), keepdim=True)

        loss_fn.update_center(teacher_out)

        expected_center = (torch.zeros_like(batch_center) * momentum) + batch_center * (1 - momentum)
        self.assertTrue(torch.allclose(loss_fn.center, expected_center, atol=1e-7, rtol=0.0))

        # Second update should incorporate momentum
        teacher_out2 = torch.randn(2, 3, 4)
        batch_center2 = teacher_out2.mean(dim=(0, 1), keepdim=True)
        prev = loss_fn.center.clone()

        loss_fn.update_center(teacher_out2)
        expected2 = prev * momentum + batch_center2 * (1 - momentum)
        self.assertTrue(torch.allclose(loss_fn.center, expected2, atol=1e-7, rtol=0.0))

    def test_forward_updates_center_and_returns_scalar(self):
        loss_fn = self._make_loss(output_dim=4, center_momentum=0.5)
        teacher_out = torch.randn(2, 2, 4)
        student_out = torch.randn(3, 2, 4)

        center_before = loss_fn.center.clone()
        loss = loss_fn(teacher_out, student_out, epoch=None)

        self.assertEqual(loss.dim(), 0)  # scalar tensor
        self.assertTrue(torch.isfinite(loss).item())

        # center must change (very likely); verify it equals the update_center formula exactly
        batch_center = teacher_out.mean(dim=(0, 1), keepdim=True)
        expected_center = center_before * 0.5 + batch_center * 0.5
        self.assertTrue(torch.allclose(loss_fn.center, expected_center, atol=1e-7, rtol=0.0))

    def test_gradients_flow_to_student_and_not_to_center(self):
        loss_fn = self._make_loss(output_dim=4)
        loss_fn.center.zero_()

        teacher_out = torch.randn(2, 2, 4, requires_grad=True)
        student_out = torch.randn(3, 2, 4, requires_grad=True)

        loss = loss_fn(teacher_out, student_out, epoch=None)
        loss.backward()
        self.assertIsNotNone(student_out.grad)
        self.assertTrue(torch.isfinite(student_out.grad).all().item())
        self.assertGreater(float(student_out.grad.abs().sum()), 0.0)

        # Center is a buffer and should not accumulate grad
        self.assertFalse(loss_fn.center.requires_grad)
        self.assertIsNone(loss_fn.center.grad)


class TestBetaVAELoss(unittest.TestCase):
    """ Test the Beta-VAE loss.
    """
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.rand(4, 3)  # batch of 4, 3 features
        self.mu = torch.zeros_like(self.x)
        self.std = torch.ones_like(self.x)
        self.q = Normal(self.mu, self.std)

    def test_invalid_default_dist(self):
        with self.assertRaises(ValueError):
            BetaVAELoss(default_dist="foo")

    def test_valid_default_dist(self):
        for dist in ["normal", "laplace", "bernoulli"]:
            loss_fn = BetaVAELoss(default_dist=dist)
            self.assertIsInstance(loss_fn, BetaVAELoss)

    def test_reconstruction_normal(self):
        p = Normal(self.x, torch.ones_like(self.x))
        loss_fn = BetaVAELoss(beta=1.0, default_dist="normal")
        loss = loss_fn.reconstruction_loss(p, self.x)
        self.assertGreaterEqual(loss.item(), 0)

    def test_reconstruction_laplace(self):
        p = Laplace(self.x, torch.ones_like(self.x))
        loss_fn = BetaVAELoss(beta=1.0, default_dist="laplace")
        loss = loss_fn.reconstruction_loss(p, self.x)
        self.assertGreaterEqual(loss.item(), 0)

    def test_reconstruction_bernoulli(self):
        probs = torch.sigmoid(self.x)
        p = Bernoulli(probs=probs)
        loss_fn = BetaVAELoss(beta=1.0, default_dist="bernoulli")
        loss = loss_fn.reconstruction_loss(p, probs)
        self.assertGreaterEqual(loss.item(), 0)

    def test_reconstruction_unknown_distribution(self):
        loss_fn = BetaVAELoss()
        with self.assertRaises(TypeError):
            loss_fn.reconstruction_loss("foo", self.x)

    def test_kl_divergence_zero_for_standard_normal(self):
        q = Normal(torch.zeros_like(self.x), torch.ones_like(self.x))
        loss_fn = BetaVAELoss()
        kl = loss_fn.kl_normal_loss(q)
        self.assertAlmostEqual(kl.item(), 0.0, places=5)

    def test_kl_divergence_positive(self):
        q = Normal(2 * torch.ones_like(self.x), torch.ones_like(self.x))
        loss_fn = BetaVAELoss()
        kl = loss_fn.kl_normal_loss(q)
        self.assertGreater(kl.item(), 0.0)

    def test_call_with_distribution(self):
        p = Normal(self.x, torch.ones_like(self.x))
        loss_fn = BetaVAELoss(beta=2.0)
        out = loss_fn(self.x, p, self.q)
        self.assertIn("loss", out)
        self.assertIn("rec_loss", out)
        self.assertIn("kl_loss", out)
        self.assertAlmostEqual(
            out["loss"].item(),
            out["rec_loss"].item() + 2.0 * out["kl_loss"].item(),
            places=5,
        )

    def test_call_with_tensor_as_p(self):
        p_mean = self.x.clone()
        loss_fn = BetaVAELoss(default_dist="normal")
        out = loss_fn(self.x, p_mean, self.q)
        self.assertIsInstance(out["loss"], torch.Tensor)

    def test_parse_tensor_for_each_default(self):
        for dist in ["normal", "laplace", "bernoulli"]:
            loss_fn = BetaVAELoss(default_dist=dist)
            p = loss_fn._parse_distribution(self.x)
            self.assertTrue(
                isinstance(p, (Normal, Laplace, Bernoulli)),
                f"Expected a distribution for {dist}",
            )

    def test_parse_invalid_type(self):
        loss_fn = BetaVAELoss()
        with self.assertRaises(TypeError):
            loss_fn._parse_distribution(123)  # not a tensor or distribution


if __name__ == "__main__":
    unittest.main()
