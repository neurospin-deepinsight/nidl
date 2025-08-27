##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.distributions import Bernoulli, Categorical, Normal

from ...losses import MCVAELoss
from ..base import BaseEstimator, ClassifierMixin
from .vae import VAE


class MCVAE(ClassifierMixin, BaseEstimator):
    """ Multi-Channel Variational Autoencoder for the joint analysis of
    heterogeneous data.

    The MCVAE is an extension of the VAE able to jointly model multiple data
    source that are named channels.

    The sparse MCVAE (sMCVAE) is an MCVAE along with an option to have a sparse
    latent representation.

    This estimator automatically instanciate :class:`~nidl.estimators.ae.VAE`
    using the specified `vae_kwargs` parameters.

    [1] Sparse Multi-Channel Variational Autoencoder for the Joint Analysis of
    Heterogeneous Data, Antelmi, Luigi, PMLR 2019.

    Parameters
    ----------
    lr: float
        the learning rate.
    weight_decay: float
        the Adam optimizer weight decay parameter.
    latent_dim: int
        the number of latent dimensions.
    n_feats: list of int
        each channel input dimensions.
    sparse: bool, default=False
        use sparsity contraint.
    beta: float, default=1.
        scaling factor for Kullback-Leibler distance (beta-VAE).
    noise_init_logvar: float, default=-3
        default noise parameters values.
    noise_fixed: bool, default=False
        False to skip gradients on noise parameters.
    vae_kwargs: dict, default=None
        extra parameters passed initialization of the VAE model.
    dropout_threshold: float, default=None
        dropout threshold.
    random_state: int, default=None
        setting a seed for reproducibility.
    kwargs: dict
        trainer parameters.

    Attributes
    ----------
    model
        a :class:`~torch.nn.Module` containing the prediction model.
    return_reconstructions
        False when the predict step needs to return the embeddings rather than
        the reconstruction.
        Default True.

    Raises
    ------
    AttributeError
        if the dropout rate does not lie in ]0, 1] or is not specified is
        sparse MCVAE.
    NotImplementedError
        if you try to apply a dropout threshold on a MCVAE.
    """
    def __init__(
            self,
            lr: float,
            weight_decay: float,
            latent_dim: int,
            n_feats: list[int],
            sparse: bool = False,
            beta: float = 1.,
            noise_init_logvar: int = -3,
            noise_fixed: bool = False,
            vae_model: str = "dense",
            vae_kwargs: Optional[dict] = None,
            dropout_threshold: Optional[float] = None,
            random_state: Optional[int] = None,
            **kwargs):
        if (dropout_threshold is not None and (
                dropout_threshold <= 0 or dropout_threshold > 1)):
            raise AttributeError(
                "The dropout rate must lie in ]0, 1]."
            )
        if sparse and dropout_threshold is None:
            raise AttributeError(
                "You need to specify the droput rate is a sMCVAE."
            )
        super().__init__(random_state=random_state, **kwargs)
        self.latent_dim = latent_dim
        self.n_feats = n_feats
        self.n_channels = len(n_feats)
        self.sparse = sparse
        self.beta = beta
        self.noise_init_logvar = noise_init_logvar
        self.noise_fixed = noise_fixed
        self.vae_kwargs = vae_kwargs or {}
        self.dropout_threshold = dropout_threshold
        self.return_reconstructions_ = True
        self.init_vae()
        self.criterion = MCVAELoss(
            self.n_channels, beta=self.beta, sparse=self.sparse)

    @property
    def return_reconstructions(self):
        return self.return_reconstructions_

    @return_reconstructions.setter
    def return_reconstructions(
            self,
            val: bool):
        self.return_reconstructions_ = val

    @property
    def dropout(self):
        if self.sparse:
            alpha = torch.exp(self.log_alpha.detach().cpu())
            return alpha / (alpha + 1)
        else:
            raise NotImplementedError(
                "Dropout property only available in sparse mode.")

    def init_vae(self):
        """ Create one VAE model per channel.
        """
        if self.sparse:
            self.log_alpha = nn.Parameter(
                torch.FloatTensor(1, self.latent_dim).normal_(0, 0.01))
        else:
            self.log_alpha = None
        vae = []
        for c_idx in range(self.n_channels):
            if "conv_flts" not in self.vae_kwargs:
                self.vae_kwargs["conv_flts"] = None
            if "dense_hidden_dims" not in self.vae_kwargs:
                self.vae_kwargs["dense_hidden_dims"] = None
            vae.append(
                VAE(
                    input_channels=1,
                    input_dim=self.n_feats[c_idx],
                    latent_dim=self.latent_dim,
                    noise_out_logvar=self.noise_init_logvar,
                    noise_fixed=self.noise_fixed,
                    sparse=self.sparse,
                    act_func=torch.nn.LeakyReLU,
                    final_activation=False,
                    log_alpha=self.log_alpha,
                    **self.vae_kwargs,
                )
            )
        self.vae = torch.nn.ModuleList(vae)

    def encode(
            self,
            x: Sequence[torch.Tensor]) -> Sequence[Normal]:
        """ Encodes the input by passing through the encoder network
        and returns the latent distribution for each channel.

        Parameters
        ----------
        x: list of Tensor, (C,) -> (N, Fc)
            input tensors to encode.

        Returns
        -------
        out: list of Normal (C,) -> (N, D)
            each channel distribution parameters mu (mean of the latent
            Gaussian) and logvar (standard deviation of the latent Gaussian).
        """
        return [self.vae[c_idx].encode(x[c_idx])
                for c_idx in range(self.n_channels)]

    def decode(
            self,
            z: Sequence[torch.Tensor]) -> Sequence[Normal]:
        """ Maps the given latent codes onto the image space.

        Parameters
        ----------
        z: list of Tensor (N, D)
            sample from the distribution having latent parameters mu, var.

        Returns
        -------
        p: list of Normal, (N, C, F)
            the prediction p(x|z).
        """
        p = []
        for c_idx1 in range(self.n_channels):
            pi = [self.vae[c_idx1].decode(z[c_idx2])
                  for c_idx2 in range(self.n_channels)]
            p.append(pi)
            del pi
        return p

    def reconstruct(
            self,
            p: Sequence[Normal]) -> Sequence[np.ndarray]:
        """ Get the reconstructed data from the prediction p(x|z).
        """
        x_hat = []
        for c_idx1 in range(self.n_channels):
            x_tmp = torch.stack([
                p[c_idx1][c_idx2].loc.detach()
                for c_idx2 in range(self.n_channels)]).mean(dim=0)
            x_hat.append(x_tmp.cpu().numpy())
            del x_tmp
        return x_hat

    @classmethod
    def p_to_prediction(
            cls,
            p: Union[list, tuple, Normal, Categorical, Bernoulli]):
        """ Get the prediction from various types of distributions.
        """
        if isinstance(p, (list, tuple)):
            return [cls.p_to_prediction(_p) for _p in p]
        elif isinstance(p, Normal):
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

    def apply_threshold(
            self,
            z: Tensor,
            keep_dims: bool = True,
            reorder: bool = False,
            ndim: Optional[int] = None) -> list:
        """ Apply dropout threshold.

        Parameters
        ----------
        z: Tensor
            distribution samples.
        keep_dims: bool default True
            dropout lower than threshold is set to 0.
        reorder: bool default False
            reorder dropout rates.
        ndim: int, default None
            fix the number of dimensions to keep.

        Returns
        -------
        z_keep: list
            dropout rates.
        """
        if not self.sparse:
            raise NotImplementedError(
                "Can't apply a dropout threshold on a MCVAE."
            )
        order = torch.argsort(self.dropout).squeeze()
        keep = (self.dropout < self.dropout_threshold).squeeze()
        if ndim is not None:
            self.dropout_threshold = np.sort(self.dropout.squeeze())[ndim - 1]
        keep = (self.dropout <= self.dropout_threshold).squeeze()
        z_keep = []
        for drop in z:
            if keep_dims:
                drop[:, ~keep] = 0
            else:
                drop = drop[:, keep]
                order = torch.argsort(
                    self.dropout[self.dropout < self.dropout_threshold])
                order = order.squeeze()
            if reorder:
                drop = drop[:, order]
            z_keep.append(drop)
            del drop
        return z_keep

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

    def mcvae_loss(
            self,
            batch: Sequence[torch.Tensor],
            mode: str):
        # Encode all images
        qs = self.encode(batch)
        z = [q.rsample() for q in qs]
        p = self.decode(z)
        # Calculate loss
        self.criterion.layer_outputs = {"q": qs, "x": batch}
        loss, extra_loss = self.criterion(p)
        # Logging loss
        self.log(mode + "_loss", loss, prog_bar=True)
        if mode == "train":
            for key, val in extra_loss.items():
                self.log(mode + f"_{key}", val, prog_bar=True)
            if self.log_alpha is not None:
                do = np.sort(self.dropout.numpy().reshape(-1))
                self.log("dropout_rate", do.mean(), prog_bar=True)
        return loss, extra_loss, qs

    def training_step(
            self,
            batch: Sequence[torch.Tensor],
            batch_idx: int,
            dataloader_idx: Optional[int] = 0):
        loss, extra_loss, _ = self.mcvae_loss(batch, mode="train")
        return loss

    def validation_step(
            self,
            batch: Sequence[torch.Tensor],
            batch_idx: int,
            dataloader_idx: Optional[int] = 0):
        loss, extra_loss, qs = self.mcvae_loss(batch, mode="val")
        return qs

    def predict_step(
            self,
            batch: Sequence[torch.Tensor],
            batch_idx: int,
            dataloader_idx: Optional[int] = 0):
        qs = self.encode(batch)
        z = [q.loc for q in qs]
        if self.sparse:
            z = self.apply_threshold(z)
        if self.return_reconstructions_:
            p = self.decode(z)
            return p
        else:
            return z
