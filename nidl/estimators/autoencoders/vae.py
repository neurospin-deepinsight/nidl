##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

from typing import Optional

import torch
from torch import nn, optim

from ...losses import BetaVAELoss
from ..base import BaseEstimator, TransformerMixin


class VAE(TransformerMixin, BaseEstimator):
    r"""Variational Auto-Encoder (VAE) [1]_ [2]_.

    See Also: :class:`~nidl.losses.beta_vae.BetaVAELoss`

    A VAE is a probabilistic generative model that learns a latent
    representation of input data and reconstructs it. It implements `fit` and
    `transform` methods to respectively train the model and obtain the latent
    embeddings.

    The VAE consists of three main components:

    - Encoder: maps input `x` to latent mean :math:`\mu` and log-variance
      :math:`\log \sigma^2`
    - Reparameterization trick: samples latent vector
      :math:`z \sim q(z | x) = \mathcal{N}(\mu, \sigma^2 I)`
    - Decoder: reconstructs input :math:`\hat{x}` from latent vector :math:`z`

    The model is trained by minimizing the sum of two components:

    - **Reconstruction loss**: Measures how well the decoder reconstructs the
      input.

        * For binary data: Binary Cross-Entropy (BCE) loss
        * For continuous data: Mean Squared Error (MSE) loss

        .. math::
            \mathcal{L}_{recon} = - \mathbb{E}_{q(z|x)} [ \log p(x|z) ]
    - **KL Divergence loss**: Encourages the latent distribution :math:`q(z|x)`
      to be close to the prior :math:`p(z) = \mathcal{N}(0, I)`.

        .. math::
            \mathcal{L}_{KL} = D_{KL}(q(z|x) | p(z))

    The total loss is a weighted sum of these two components:

    .. math::
        \mathcal{L}_{total} = \mathcal{L}_{recon} + \beta \mathcal{L}_{KL}

    Parameters
    ----------
    encoder: class:`~torch.nn.Module`
        The encoder mapping input ``x`` to the representation space. The mean
        :math:`\mu` and log-variance  :math:`\log \sigma^2` layers
        are automatically added according to the ``latent_dim`` parameter.
    decoder: class:`~torch.nn.Module`
        The decoder backbone outputting :math:`p(x | z)` as a
        `torch.distributions` or a `torch.Tensor` representing the mean of a
        Normal (default) or Laplace distribution.
    encoder_out_dim: int
        The output size of the encoder.
    latent_dim: int
        The number of latent dimensions (which is the size of the mean and
        variance of the posterior distribution).
    beta: float, default=1.
        Scaling factor for Kullback-Leibler distance (beta-VAE).
    default_dist: str, default="normal"
        Default decoder distribution. It defines the reconstruction loss
        (L2 for Normal, L1 for Laplace, cross-entropy for Bernoulli).
    stochastic_transform: bool, default=True
        If True (default), the transformed data are obtained by sampling
        according to the posterior distribution :math:`q(z | x)`.If False,
        the mean of the posterior distribution is returned.
    lr: float
        the learning rate.
    weight_decay: float
        the Adam optimizer weight decay parameter.
    random_state: int, default=None
        setting a seed for reproducibility.
    kwargs: dict
        trainer parameters.

    Attributes
    ----------
    encoder: :class:`~torch.nn.Module`
        The encoder network.
    decoder: :class:`~torch.nn.Module`
        The decoder network.
    fc_mu: :class:`~torch.nn.Module`
        The linear layer mapping the encoder output to the mean :math:`\mu` of
        the posterior distribution.
    fc_logvar: :class:`~torch.nn.Module`
        The linear layer mapping the encoder output to the log-variance
        :math:`\log \sigma^2` of the posterior distribution.

    References
    ----------
    .. [1] Diederik P. Kingma, Max Welling, "Auto-Encoding Variational Bayes",
       ICLR 2014.
    .. [2] Irina Higgins et al., "beta-VAE: Learning Basic Visual Concepts with
       a Constrained Variational Framework", ICLR 2017.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encoder_out_dim: int,
        latent_dim: int,
        beta: float = 1.0,
        default_dist: str = "normal",
        stochastic_transform: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            random_state=random_state,
            ignore=["callbacks", "encoder", "decoder"],
            **kwargs,
        )
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_out_dim = encoder_out_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.stochastic_transform = stochastic_transform
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = BetaVAELoss(beta=self.beta, default_dist=default_dist)

        self.fc_mu = nn.Linear(self.encoder_out_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_out_dim, self.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input and sample from the posterior distribution q(z|x).

        Parameters
        ----------
        x: torch.Tensor
            Input data given to the encoder.

        Returns
        -------
        z: torch.Tensor, shape (batch_size, latent_dim)
            Latent vector sampled from the posterior distribution.
        """
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        _, z = self._sample(mu, log_var)
        return z

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        """Perform one training step and computes and logs training losses.

        Three losses are logged: the beta-VAE loss ("loss"), the reconstruction
        loss ("rec_loss") and the KL divergence loss ("kl_loss").

        Parameters
        ----------
        batch: torch.Tensor
            The input data given to the encoder.
        batch_idx: int
            Ignored.
        dataloader_idx: Optional[int], default=0
            Ignored.

        Returns
        -------
        losses: dict
            Dictionary with "loss", "rec_loss", "kl_loss" as keys.
        """
        x = batch
        p, q = self._run_step(x)
        losses = self.criterion(x, p, q)
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            on_step=True,
            on_epoch=False,
        )
        return losses

    def validation_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        """Perform one validation step and computes and logs validation
        losses.

        Three losses are logged: the beta-VAE loss ("loss"), the reconstruction
        loss ("rec_loss") and the KL divergence loss ("kl_loss").

        Parameters
        ----------
        batch: torch.Tensor
            The input data given to the encoder.
        batch_idx: int
            Ignored.
        dataloader_idx: Optional[int], default=0
            Ignored.

        Returns
        -------
        losses: dict
            Dictionary with "loss", "rec_loss", "kl_loss" as keys.
        """
        x = batch
        p, q = self._run_step(x)
        losses = self.criterion(x, p, q)
        self.log_dict({f"val/{k}": v for k, v in losses.items()})
        return losses

    def transform_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        """Transform the input data to the latent space.

        By default, the latent vector is obtained by sampling according to the
        posterior distribution :math:`q(z | x)`. It is just the mean of the
        distribution if ``stochastic_transform`` is False.

        Parameters
        ----------
        batch: torch.Tensor
            The input data given to the encoder.
        batch_idx: int
            Ignored.
        dataloader_idx: Optional[int], default=0
            Ignored.

        Returns
        -------
        z: torch.Tensor, shape (batch_size, latent_dim)
            The latent vector.
        """
        x = self.encoder(batch)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        if self.stochastic_transform:
            _, z = self._sample(mu, log_var)
            return z
        return mu

    def configure_optimizers(self):
        """Declare an :class:`~torch.optim.AdamW` optimizer."""
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return [optimizer]

    def sample(self, n_samples):
        """Generate `n_samples` by sampling from the latent space.

        Parameters
        ----------
        nsamples: int
            Number of samples to generate.

        Returns
        -------
        x: torch.Tensor
            Generated samples.
        """
        z = torch.randn(n_samples, self.latent_dim)
        return self.decoder(z)

    def _run_step(self, x: torch.Tensor):
        """Encode the input and sample from the posterior distribution q(z|x).

        Parameters
        ----------
        x: torch.Tensor
            Input data given to the encoder.

        Returns
        -------
        z: torch.Tensor, shape (batch_size, latent_dim)
            Latent vector sampled from the posterior distribution.
        q: torch.distributions
            Probabilistic encoder.
        """
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        q, z = self._sample(mu, log_var)
        return self.decoder(z), q

    def _sample(self, mu: torch.Tensor, log_var: torch.Tensor):
        """Reparameterization trick: samples latent vector."""
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return q, z
