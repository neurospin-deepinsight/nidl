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
    InfoNCE,
    YAwareInfoNCE,
    KernelMetric,
    BetaVAELoss,
    DCL,
    DCLW,
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

    def test_dcl_bad_parameters(self):
        """Test ValueError is raised when bad values of parameters
        are used to initialise DCL
        """
        with self.assertRaises(ValueError):
            DCL(pos_weight_fn='string_fn')
            DCL(temperature=-0.1)
    
    def test_dclw_init(self):
        """Test the initialisation of the weighting function in DCLW.
        """
        loss = DCLW(sigma=0.5, temperature=0.2)
        assert hasattr(loss, "pos_weight_fn")
        assert callable(loss.pos_weight_fn)
        assert loss.temperature == 0.2

    def test_dcl_dclw_forward(self):
        """Test the computation of the loss in DCL and DCLW.
        """
        for temperature in [0.1, 1.0, 5.0]:
            for batch_size in [1, 10]:
                for n_embedding in [1, 10]:
                    dcl = DCL(temperature=temperature, pos_weight_fn=None)
                    dclw = DCLW(sigma=temperature, temperature=temperature)
                    for loss in [dcl, dclw]:
                        z1 = torch.rand(batch_size, n_embedding)
                        z2 = torch.rand(batch_size, n_embedding)
                        # Perfect alignment
                        loss_low = loss(z1, z1)
                        # random alignment
                        loss_high = loss(z1, z2)
                        assert loss_low <= loss_high, (
                            f"{str(loss)} loss should be lower for aligned embeddings, "
                            f"got {loss_low} vs {loss_high}"
                        )
                        # Assert loss is correct type
                        assert isinstance(loss_high, torch.Tensor)
                        assert loss_high.dim() == 0   # scalar loss
                        assert not torch.isnan(loss_high)
                        assert not torch.isinf(loss_high)

    def test_dcl_dclw_backward(self):
        '''Test DCL and DCLW are differentiable.
        '''
                # Check that the loss is differentiable
        z1 = torch.randn(4, 5, requires_grad=True)
        z2 = torch.randn(4, 5, requires_grad=True)
        temperature = 0.1

        for loss_fn in [DCL(temperature=temperature, pos_weight_fn=None),
                        DCLW(sigma=temperature, temperature=temperature)]:
            loss = loss_fn(z1, z2)
            loss.backward()

            self.assertIsNotNone(z1.grad)
            self.assertIsNotNone(z2.grad)
            self.assertEqual(z1.grad.shape, z1.shape)
            self.assertEqual(z2.grad.shape, z2.shape)


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
