##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from typing import Optional, Union

import torch
import torch.optim as optim
from torch.distributions import Bernoulli, Categorical, Normal

from ...losses import BetaHLoss
from ...volume.backbones import VAE as _VAE
from ..base import BaseEstimator, TransformerMixin


class VAE(TransformerMixin, BaseEstimator):
    """ Variational Auto-Encoder (VAE).

    [1] Sparse Multi-Channel Variational Autoencoder for the Joint Analysis of
    Heterogeneous Data, Antelmi, Luigi, PMLR 2019,
    https://github.com/ggbioing/mcvae.

    Parameters
    ----------
    lr: float
        the learning rate.
    weight_decay: float
        the Adam optimizer weight decay parameter.
    latent_dim: int
        the number of latent dimensions.
    n_feat: int
        the input dimensions.
    beta: float, default=1.
        scaling factor for Kullback-Leibler distance (beta-VAE).
    vae_kwargs: dict, default=None
        extra parameters passed initialization of the VAE model.
    random_state: int, default=None
        setting a seed for reproducibility.
    kwargs: dict
        trainer parameters.

    Attributes
    ----------
    vae
        a :class:`~torch.nn.Module` containing the transformer model.
    return_reconstructions
        False when the predict step needs to return the embeddings rather than
        the reconstruction.
        Default True.
    """
    def __init__(
            self,
            lr: float,
            weight_decay: float,
            latent_dim: int,
            n_feat: int,
            beta: float = 1.,
            vae_model: str = "dense",
            vae_kwargs: Optional[dict] = None,
            random_state: Optional[int] = None,
            **kwargs):
        super().__init__(random_state=random_state, **kwargs)
        self.latent_dim = latent_dim
        self.n_feat = n_feat
        self.beta = beta
        self.vae_kwargs = vae_kwargs or {}
        self.init_vae()
        self.criterion = BetaHLoss(beta=self.beta)

    def init_vae(self):
        """ Create one VAE model per channel.
        """
        if "conv_flts" not in self.vae_kwargs:
            self.vae_kwargs["conv_flts"] = None
        if "dense_hidden_dims" not in self.vae_kwargs:
            self.vae_kwargs["dense_hidden_dims"] = None
        self.vae = _VAE(
            input_channels=1,
            input_dim=self.n_feat,
            latent_dim=self.latent_dim,
            noise_out_logvar=self.noise_init_logvar,
            noise_fixed=self.noise_fixed,
            sparse=self.sparse,
            act_func=torch.nn.LeakyReLU,
            final_activation=False,
            log_alpha=self.log_alpha,
            **self.vae_kwargs,
        )

    @classmethod
    def p_to_prediction(
            cls,
            p: Union[list, tuple, Normal, Categorical, Bernoulli]):
        """ Get the prediction from various types of distributions.
        """
        if isinstance(p, Normal):
            pred = p.loc
        elif isinstance(p, Categorical):
            pred = p.logits.argmax(dim=1)
        elif isinstance(p, Bernoulli):
            pred = p.probs
        else:
            raise NotImplementedError(
                f"Disribution type {p.__class__.__name__} not yet supported."
            )
        return pred.cpu().detach().numpy()

    def configure_optimizers(self):
        """ Declare a :class:`~torch.optim.AdamW` optimizer and, optionnaly
        (``max_epochs`` is defined), a
        :class:`~torch.optim.lr_scheduler.MultiStepLR` learning-rate
        scheduler.
        """
        optimizer = optim.AdamW(
            self.parameters(), lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay)
        if (hasattr(self.hparams, "max_epochs") and
                self.hparams.max_epochs is not None):
            lr_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[int(self.hparams.max_epochs * 0.6),
                int(self.hparams.max_epochs * 0.8)], gamma=0.1)
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]

    def vae_loss(
            self,
            batch: torch.Tensor,
            mode: str):
        # Encode all images
        qs = self.vae.encode(batch)
        z = qs.rsample()
        p = self.vae.decode(z)
        # Calculate loss
        self.criterion.layer_outputs = {"q": qs, "x": batch, "model": self.vae}
        loss, extra_loss = self.criterion(p, batch)
        # Logging loss
        self.log(mode + "_loss", loss, prog_bar=True)
        if mode == "train":
            for key, val in extra_loss.items():
                self.log(mode + f"_{key}", val, prog_bar=True)
        return loss, extra_loss, qs

    def training_step(
            self,
            batch: torch.Tensor,
            batch_idx: int,
            dataloader_idx: Optional[int] = 0):
        loss, extra_loss, _ = self.vae_loss(batch, mode="train")
        return loss

    def validation_step(
            self,
            batch: torch.Tensor,
            batch_idx: int,
            dataloader_idx: Optional[int] = 0):
        loss, extra_loss, qs = self.vae_loss(batch, mode="val")
        return qs

    def transform_step(
            self,
            batch: torch.Tensor,
            batch_idx: int,
            dataloader_idx: Optional[int] = 0):
        qs = self.encode(batch)
        return self.p_to_prediction(qs)
