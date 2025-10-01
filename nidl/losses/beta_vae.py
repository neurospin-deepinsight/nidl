##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import torch
from torch.distributions import Bernoulli, Laplace, Normal, kl_divergence
from torch.nn import functional as func


class BetaVAELoss:
    r""" Compute the Beta-VAE loss [1]_.

    See Also: :class:`~nidl.estimators.autoencoders.vae.VAE`

    The Beta-VAE was introduced to learn disentangled representations and
    improve interpretability. The idea is to keep the distance between the
    real and estimated posterior distribution small (under a small constant
    delta) while maximizing the probability of generating real data:

    .. math::
        \underset{\phi, \theta}{\mathrm{max}}
            \underset{x \sim D}{\mathbb{E}}\left[
                \underset{z \sim q_\phi(z | x)}{\mathbb{E}}
                    log \ p_\theta(x|z)
            \right] \\
        \text{subject to} D_{KL}(q_\phi(z|x) | p_\theta(z)) < \delta

    We can rewrite this equation as a Lagrangian with a Lagrangian multiplier
    :math:`\beta`, which leads to the Beta-VAE loss function:

    .. math::
        L_{VAE}(\theta, \phi) = L_{MSE}(\theta, \phi) -
                                  \beta L_{KL}(\theta, \phi)

    When :math:`\beta=1`, it corresponds to a VAE loss. If :math:`\beta>1`,
    this puts more weight on statistical independence than on reconstruction.
    Note that such a stronger constraint on the latent bottleneck limits
    the representation capacity of z.

    Parameters
    ----------
    beta: float, default=4.0
        Weight of the KL divergence.
    default_dist: {"normal", "laplace", "bernoulli"}, default="normal"
        Default decoder distribution. It defines the reconstruction loss
        (L2 for Normal, L1 for Laplace, cross-entropy for Bernoulli).

    Raises
    ------
    ValueError
        If the input distribution is not supported.

    References
    ----------
    .. [1] Irina Higgins et al., "beta-VAE: Learning Basic Visual Concepts with
       a Constrained Variational Framework", ICLR 2017.
    """
    def __init__(
            self, beta: float = 4.0,
            default_dist: str = "normal"):
        self.beta = beta
        self.default_dist = default_dist
        if default_dist not in {"normal", "laplace", "bernoulli"}:
            raise ValueError(
                "Default decoder distribution must be 'normal', 'laplace' or "
                f"'bernouilli', got {default_dist}"
            )

    def __call__(self, x, p, q):
        """ Compute the loss.

        Parameters
        ----------
        x: torch.Tensor
            The input data.
        p: torch.distributions or torch.Tensor
            Decoder distribution :math:`p(x | z)` for a given latent code `z`
            if `p` is :class:`torch.distributions`. If `p` is `torch.Tensor`,
            it should be the distribution mean for Normal or Laplacian
            distribution or probability of success for Bernouilli
            distribution.
        q: torch.distributions
            Probabilistic encoder (or estimated posterior probability
            function).

        Returns
        ----------
        losses: dict
            Dictionary containing the beta-VAE loss ("loss") along with all
            composite terms: the reconstruction loss "rec_loss" and KL loss
            "kl_loss".
        """
        p = self._parse_distribution(p)
        rec_loss = self.reconstruction_loss(p, x)
        kl_loss = self.kl_normal_loss(q)
        loss = rec_loss + self.beta * kl_loss
        return {"rec_loss": rec_loss, "kl_loss": kl_loss, "loss": loss}

    def reconstruction_loss(self, p, data):
        """ Computes the per image reconstruction loss for a batch of data
        (i.e. negative log likelihood).

        The distribution of the likelihood on the each pixel implicitely
        defines the loss. Bernoulli corresponds to a binary cross entropy.
        Gaussian distribution corresponds to MSE. Laplace distribution
        corresponds to L1.

        Parameters
        ----------
        p: torch.distributions
            probabilistic decoder (or likelihood of generating true data
            sample given the latent code).
        data: torch.Tensor
            The observed data.

        Returns
        -------
        loss: torch.Tensor
            per image cross entropy (i.e. normalized per batch but not pixel
            and channel).
        """
        if isinstance(p, Bernoulli):
            loss = func.binary_cross_entropy(p.probs, data, reduction="sum")
        elif isinstance(p, Normal):
            loss = func.mse_loss(p.loc, data, reduction="sum")
        elif isinstance(p, Laplace):
            loss = func.l1_loss(p.loc, data, reduction="sum")
            loss = loss * (loss != 0)  # masking to avoid nan
        else:
            raise ValueError(f"Unkown distribution: {p}")

        batch_size = len(data)
        loss = loss / batch_size

        return loss

    def kl_normal_loss(self, q):
        """ Computes the KL divergence between a normal distribution
        with diagonal covariance and a unit normal distribution.

        Parameters
        ----------
        q: torch.distributions
            probabilistic encoder (or estimated posterior probability
            function).
        """
        dimension_wise_kl = kl_divergence(q, Normal(0, 1)).mean(dim=0)
        return dimension_wise_kl.sum()

    def _parse_distribution(self, p):
        """ Check that the input parameter is a valid distribution.

        If a tensor is given, it will be converted automatically to the
        specified default distributon.

        Parameters
        ----------
        p: torch.distributions or torch.Tensor
            the input distribution.

        Returns
        -------
        p: torch.distributions
            the returned distribution.
        """
        if isinstance(p, (Bernoulli, Normal, Laplace)):
            return p
        elif isinstance(p, torch.Tensor):
            if self.default_dist == "normal":
                p = Normal(p, torch.ones_like(p))
            elif self.default_dist == "laplace":
                p = Laplace(p, torch.ones_like(p))
            elif self.default_dist == "bernoulli":
                p = Bernoulli(probs=p)
            return p
        else:
            raise ValueError(
                "Decoder distribution must be `torch.distributions` or "
                f"`torch.Tensor`, got {type(p)}"
            )
