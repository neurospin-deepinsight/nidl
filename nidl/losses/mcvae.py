##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal, kl_divergence


class MCVAELoss(nn.Module):
    """ MCVAE loss.

    The loss can be written as beta * KL_loss + LL_loss, where:

    1. KL divergence loss: how off the distribution over the latent space is
       from the prior.
    2. log-likelihood LL

    [1] Sparse Multi-Channel Variational Autoencoder for the Joint Analysis of
    Heterogeneous Data, Antelmi, Luigi, PMLR 2019.

    Parameters
    ----------
    n_channels: int
        the number of channels.
    beta, float, default 1.
        for beta-VAE.
    enc_channels: list of int, default None
        encode only these channels (for kl computation).
    dec_channels: list of int, default None
        decode only these channels (for ll computation).
    sparse: bool, default False
        use sparsity contraint.
    nodecoding: bool, default False
        if set do not apply the decoding.
    """
    def __init__(
            self,
            n_channels: int,
            beta: float = 1.,
            enc_channels: Optional[list[int]] = None,
            dec_channels: Optional[list[int]] = None,
            sparse: bool = False,
            nodecoding: bool = False):
        super().__init__()
        self.n_channels = n_channels
        self.beta = beta
        self.sparse = sparse
        self.enc_channels = enc_channels
        self.dec_channels = dec_channels
        if enc_channels is None:
            self.enc_channels = list(range(n_channels))
        else:
            assert(len(enc_channels) <= n_channels)
        if dec_channels is None:
            self.dec_channels = list(range(n_channels))
        else:
            assert(len(dec_channels) <= n_channels)
        self.n_enc_channels = len(self.enc_channels)
        self.n_dec_channels = len(self.dec_channels)
        self.nodecoding = nodecoding
        self.layer_outputs = None

    def __call__(
            self,
            p: Sequence[Normal]):
        """ Compute loss.

        Parameters
        ----------
        p: list of Normal distributions (C,) -> (N, F)
            reconstructed channels data.

        Returns
        -------
        loss: float
            the current loss.
        extra_losses: dict
            the KL and LL losses.
        """
        if self.nodecoding:
            return -1
        if self.layer_outputs is None:
            raise ValueError(
                "This loss needs intermediate layers outputs. Please register "
                "an appropriate callback.")
        x = self.layer_outputs["x"]
        q = self.layer_outputs["q"]
        kl = self.compute_kl(q)
        kl *= self.beta
        ll = self.compute_ll(p, x)
        total = kl - ll
        return total, {"kl": kl, "ll": ll}

    def compute_kl(
            self,
            q: Sequence[Normal]):
        """ KL loss.
        """
        kl = 0
        for c_idx, qi in enumerate(q):
            if c_idx in self.enc_channels:
                kl += compute_kl(p1=qi, p2=Normal(0, 1), sparse=self.sparse)
        return kl

    def compute_ll(
            self,
            p: Sequence[Normal],
            x: Sequence[Tensor]):
        """ Log-likelihood loss.
        """
        ll = 0
        for c_idx1 in range(self.n_channels):
            for c_idx2 in range(self.n_channels):
                if c_idx1 in self.dec_channels and c_idx2 in self.enc_channels:
                    ll += compute_ll(p=p[c_idx1][c_idx2], x=x[c_idx1])
        return ll


def compute_ll(
        p: Normal,
        x: Tensor):
    """ Get the log-likelihood compatible with the distribution p.
    """
    ll = p.log_prob(x).view(len(x), -1)
    return ll.sum(-1, keepdim=True).mean(0)


def compute_kl(
        p1: Normal,
        p2: Optional[Normal] = None,
        sparse: Optional[bool] = False):
    """ Get the KL-divergence usinf sparse and non sparse VAE.
    """
    kl = _kl_log_uniform(p1) if sparse else kl_divergence(p1, p2)
    return kl.sum(-1, keepdim=True).mean(0)


def _kl_log_uniform(
        p: Normal):
    """ Gaussian dropout arises from the application of the local
    reparameterization trick. The prior on the encoder weight consistent
    with the optimization of the lower bound is the improper log-scale uniform.
    With this prior, the DKL of the dropout posterior can be numerically
    approximate, and learned through the optimization of the lower bound via
    gradient-based method. See paragraph 4.2 from.

    [1] Variational Dropout Sparsifies Deep Neural Networks, Molchanov, Dmitry,
    arxiv 2017
    https://arxiv.org/abs/1701.05369
    https://github.com/senya-ashukha/variational-dropout-sparsifies-dnn/
    blob/master/KL%20approximation.ipynb
    """
    mu = p.loc
    logvar = p.scale.pow(2).log()
    log_alpha = _compute_log_alpha(mu, logvar)
    k1, k2, k3 = 0.63576, 1.8732, 1.48695
    neg_kl = (k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 *
              torch.log1p(torch.exp(-log_alpha)) - k1)
    return - neg_kl


def _compute_log_alpha(mu, logvar):
    """ Clamp log alpha computation because dropout rate p in 0-99%,
    where p = alpha/(alpha+1).
    """
    return (logvar - 2 * torch.log(torch.abs(mu) + 1e-8)).clamp(
        min=-8, max=8)
