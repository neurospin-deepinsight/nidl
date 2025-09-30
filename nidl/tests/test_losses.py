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
    InfoNCE,
    YAwareInfoNCE,
    KernelMetric,
    BetaVAELoss
)

class TestLosses(unittest.TestCase):
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
            with self.assertRaises(ValueError): # kernel not fitted
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
        with self.assertRaises(ValueError):
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
        with self.assertRaises(ValueError):
            loss_fn._parse_distribution(123)  # not a tensor or distribution


if __name__ == "__main__":
    unittest.main()
