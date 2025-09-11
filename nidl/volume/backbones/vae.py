##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Bernoulli, Normal


class VAE(nn.Module):
    """ Variational Auto-Encoder (VAE).

    [1] Sparse Multi-Channel Variational Autoencoder for the Joint Analysis of
    Heterogeneous Data, Antelmi, Luigi, PMLR 2019,
    https://github.com/ggbioing/mcvae.

    Parameters
    ----------
    input_channels: int
        the number of input channels.
    input_dim: int or list of int
        the size of input.
    conv_flts: list of int
        the size of convolutional filters, if None do not include
        convolutional layers.
    dense_hidden_dims: list of int
        the size of dense hidden dimensions, if None do not include dense
        hidden layers.
    latent_dim: int
        the latent dimension.
    noise_out_logvar: float, default -3
        the init output log var.
    noise_fixed: bool, default True
        estimate the the output log var.
    log_alpha: nn.Parameter, default None
        dropout probabilities estimate.
    act_func: callable, default None
        the activation function.
    final_activation: bool, default False
        apply activation function to the final layer.
    dropout: float, default 0
        define the dropout rate.
    sparse: bool, default False
        use sparsity contraint.
    encoder: nn.Module, default None
        a custom encoder.
    decoder: nn.Module, default None
        a custom decoder.
    """
    def __init__(self, input_channels, input_dim, conv_flts, dense_hidden_dims,
                 latent_dim, noise_out_logvar=-3, noise_fixed=True,
                 log_alpha=None, act_func=None, final_activation=False,
                 dropout=0, sparse=False, encoder=None, decoder=None):
        super().__init__()
        if isinstance(input_dim, tuple):
            input_dim = list(input_dim)
        self.latent_dim = latent_dim
        self.act_func = act_func
        if sparse:
            if log_alpha is None:
                self.log_alpha = nn.Parameter(
                    torch.FloatTensor(1, self.latent_dim).normal_(0, 0.01))
            else:
                self.log_alpha = log_alpha
        else:
            self.log_alpha = None
        encoder = encoder or Encoder
        decoder = decoder or Decoder
        self.encode = encoder(
            input_channels, input_dim, conv_flts, dense_hidden_dims,
            latent_dim, act_func=act_func, dropout=dropout,
            log_alpha=self.log_alpha)
        dec_conv_flts = conv_flts[::-1] if conv_flts is not None else None
        dec_dense_hidden_dims = (
            dense_hidden_dims[::-1] if dense_hidden_dims is not None else None)

        self.decode = decoder(
            latent_dim, dec_conv_flts, dec_dense_hidden_dims,
            input_channels, input_dim, noise_out_logvar=noise_out_logvar,
            noise_fixed=noise_fixed, act_func=act_func,
            final_activation=final_activation, dropout=dropout)

    def forward(self, x):
        """ The forward method.
        """
        q = self.encode(x)
        z = self.reparametrize(q)
        p = self.decode(z)
        return p, {"q": q, "z": z, "model": self}

    def reparametrize(self, q):
        """ Implement the reparametrization trick.
        """
        z = q.rsample() if self.training else q.loc
        return z

    @staticmethod
    def p_to_prediction(p):
        """ Get the prediction from various types of distributions.
        """
        if isinstance(p, Normal):
            pred = p.loc.cpu().detach().numpy()
        elif isinstance(p, Bernoulli):
            pred = p.probs.cpu().detach().numpy()
        else:
            raise NotImplementedError
        return pred

    def reconstruct(self, x, sample=False, dropout_threshold=None):
        """ Reconstruct a new data from a given input with or without
        resampling.
        """
        with torch.no_grad():
            q = self.encode(x)
            posterior = q
            z = posterior.sample() if sample else posterior.loc
            if dropout_threshold is not None:
                z = self.apply_threshold(z, dropout_threshold)
            p = self.decode(z)
        return self.p_to_prediction(p)

    def generate(self, z=None, device=None):
        """ Generate a new data from a given sample or a random one.
        """
        device = device or torch.device("cpu")
        with torch.no_grad():
            if z is None:
                z = Normal(loc=torch.zeros(1, self.latent_dim),
                           scale=1).sample()
            z = z.to(device)
            p = self.decode(z)
            return self.p_to_prediction(p)

    def apply_threshold(self, z, threshold, keep_dims=True, reorder=False):
        """ Threshold the latent samples based on the estimated dropout
        probabilities.
        """
        assert(threshold <= 1.0)
        order = torch.argsort(self.dropout).squeeze()
        keep = (self.dropout < threshold).squeeze()
        if keep_dims:
            z[:, ~keep] = 0
        else:
            z = z[:, keep]
            order = torch.argsort(self.dropout[keep]).squeeze()
        if reorder:
            z = z[:, order]
        return z

    @property
    def dropout(self):
        """ Compute the dropout probabilities.
        """
        if self.log_alpha is not None:
            alpha = torch.exp(self.log_alpha.detach())
            return alpha / (alpha + 1)
        else:
            raise NotImplementedError


class Encoder(nn.Module):
    """ The encoder part of a VAE.

    Parameters
    ----------
    input_channels: int
        the number of input channels.
    input_dim: int or list of int
        the size of input.
    conv_flts: list of int
        the size of convolutional filters, if None do not include
        convolutional layers.
    dense_hidden_dims: list of int
        the size of dense hidden dimensions, if None do not include dense
        hidden layers.
    latent_dim: int
        the latent dimension.
    act_func: callable, default None
        the activation function.
    dropout: float, default 0
        define the dropout rate.
    log_alpha: nn.Parameter, default None
        inducing sparse latent representations.
    """
    def __init__(self, input_channels, input_dim, conv_flts, dense_hidden_dims,
                 latent_dim, act_func=None, dropout=0, log_alpha=None):
        super().__init__()
        self.act_func = act_func or nn.ReLU
        self.log_alpha = log_alpha
        if isinstance(input_dim, torch.Size):
            input_dim = list(input_dim)
        elif not isinstance(input_dim, list):
            input_dim = [input_dim]
        ndim = len(input_dim)
        if conv_flts is not None:
            w_conv_layers = Encoder.init_conv_layers(
                input_channels, conv_flts, self.act_func, dropout, ndim)
            self.w_conv = nn.Sequential(*w_conv_layers)
            flatten_dim = (
                conv_flts[-1] * np.prod(Encoder.final_conv_dim(
                    input_dim, kernels=[5] * (len(conv_flts) - 1) + [3],
                    paddings=[2] * len(conv_flts))[-1]))
        else:
            self.w_conv = None
            flatten_dim = input_channels * np.prod(input_dim)
        if dense_hidden_dims is not None:
            w_dense_layers = Encoder.init_dense_layers(
                flatten_dim, dense_hidden_dims, self.act_func, dropout)
            self.w_dense = nn.Sequential(*w_dense_layers)
            final_dim = dense_hidden_dims[-1]
        else:
            self.w_dense = None
            final_dim = flatten_dim
        self.w_mu = nn.Linear(final_dim, latent_dim)
        if self.log_alpha is None:
            self.w_logvar = nn.Linear(final_dim, latent_dim)

    @staticmethod
    def final_conv_dim(input_dim, kernels, paddings):
        """ Infer the size of eaxh sample after the convolutions bloc.
        """
        all_dims = [np.asarray(input_dim)]
        for kernel, padding in zip(kernels, paddings):
            all_dims.append((all_dims[-1] - kernel + 2 * padding) / 2 + 1)
        return np.asarray(all_dims).astype(int)

    @staticmethod
    def init_dense_layers(input_dim, hidden_dims, act_func, dropout,
                          final_activation=True):
        """ Create the dense layers.
        """
        layers = []
        current_dim = input_dim
        for cnt, dim in enumerate(hidden_dims):
            if not final_activation and cnt == (len(hidden_dims) - 1):
                layers.append(nn.Linear(current_dim, dim))
            else:
                layers.extend([
                    nn.Linear(current_dim, dim),
                    act_func()])
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            current_dim = dim
        return layers

    @staticmethod
    def init_conv_layers(input_channels, flts, act_func, dropout, ndim=1):
        """ Create the convolutional layers.
        """
        conv_fn = getattr(nn, f"Conv{ndim}d")
        layers = []
        current_channels = input_channels
        for cnt, n_filts in enumerate(flts):
            layers.extend([
                conv_fn(
                    current_channels, out_channels=n_filts,
                    kernel_size=(3 if (cnt == (len(flts) - 1)) else 5),
                    stride=2, padding=2),
                act_func()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_channels = n_filts
        return layers

    def forward(self, x):
        """ The forward method.
        """
        out = self.w_conv(x) if self.w_conv is not None else x
        out = torch.flatten(out, start_dim=1)
        if self.w_dense is not None:
            out = self.w_dense(out)
        z_mu = self.w_mu(out)
        if self.log_alpha is None:
            z_logvar = self.w_logvar(out)
        else:
            z_logvar = Encoder.compute_logvar(z_mu, self.log_alpha)
        return Normal(loc=z_mu, scale=torch.exp(0.5 * z_logvar))

    @staticmethod
    def compute_logvar(mu, log_alpha):
        """ Compute the log variance in case of sparsity contraints.
        """
        return log_alpha + 2 * torch.log(torch.abs(mu) + 1e-8)


class Decoder(nn.Module):
    """ The decoder part of a VAE.

    Parameters
    ----------
    latent_dim: int
        the latent size.
    conv_flts: list of int
        the size of convolutional filters, if None do not include
        convolutional layers.
    dense_hidden_dims: list of int
        the size of dense hidden dimensions, if None do not include dense
        hidden layers.
    output_channels: int
        the number of output channels.
    output_dim: int or list of int
        the size of output.
    noise_out_logvar: float, default -3
        the init output log var.
    noise_fixed: bool, default True
        estimate the the output log var.
    act_func: callable, default None
        the activation function.
    final_activation: bool, default False
        apply activation function to the final layer.
    dropout: float, default 0
        define the dropout rate.
    """
    def __init__(self, latent_dim, conv_flts, dense_hidden_dims,
                 output_channels, output_dim, noise_out_logvar=-3,
                 noise_fixed=True, act_func=None, final_activation=False,
                 dropout=0):
        super().__init__()
        self.act_func = act_func or nn.ReLU
        self.output_channels = output_channels
        self.conv_flts = conv_flts
        if isinstance(output_dim, torch.Size):
            output_dim = list(output_dim)
        elif not isinstance(output_dim, list):
            output_dim = [output_dim]
        ndim = len(output_dim)
        if conv_flts is not None:
            self.all_dims = Encoder.final_conv_dim(
                output_dim, kernels=[5] * (len(conv_flts) - 1) + [3],
                paddings=[2] * len(conv_flts))
            self.final_dim = self.all_dims[-1]
            self.all_dims = self.all_dims[::-1]
            flatten_dim = conv_flts[0] * np.prod(self.final_dim)
        else:
            self.final_dim = [-1]
            flatten_dim = output_channels * np.prod(output_dim)
        if dense_hidden_dims is None:
            dense_hidden_dims = []
        w_dense_layers = Encoder.init_dense_layers(
            latent_dim, [*dense_hidden_dims, flatten_dim], self.act_func,
            dropout,
            final_activation=not(not final_activation and conv_flts is None))
        self.w_dense = nn.Sequential(*w_dense_layers)
        if conv_flts is not None:
            self.w_dense = nn.Sequential(*w_dense_layers)
            w_conv_layers = Decoder.init_conv_layers(
                conv_flts[0], [*conv_flts[1:], output_channels], self.act_func,
                dropout, ndim, final_activation=final_activation)
            self.w_conv = nn.Sequential(*w_conv_layers)
        else:
            self.w_conv = None
        self.w_out_logvar = torch.nn.Parameter(
            data=torch.FloatTensor(
                1, output_channels, *output_dim).fill_(noise_out_logvar),
            requires_grad=(not noise_fixed))

    @staticmethod
    def init_conv_layers(input_channels, flts, act_func, dropout, ndim=1,
                         final_activation=True):
        """ Create the convolutional layers.
        """
        convt_fn = getattr(nn, f"ConvTranspose{ndim}d")
        layers = []
        current_channels = input_channels
        for cnt, n_flts in enumerate(flts):
            if not final_activation and cnt == (len(flts) - 1):
                layers.append(convt_fn(
                    current_channels, out_channels=n_flts,
                    kernel_size=(3 if (cnt == 0) else 5),
                    stride=2, padding=2))
            else:
                layers.extend([
                    convt_fn(
                        current_channels, out_channels=n_flts,
                        kernel_size=(3 if (cnt == 0) else 5),
                        stride=2, padding=2),
                    act_func()])
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            current_channels = n_flts
        return layers

    def forward(self, z):
        """ The forward method.
        """
        out = self.w_dense(z)
        if self.w_conv is not None:
            out = out.view(out.size(0), self.conv_flts[0], *self.final_dim)
            idx_layer = 0
            for module in self.w_conv:
                out = module(out)
                # Restore orig tensor size (preserve autoencoder structure)
                if isinstance(module, nn.modules.conv._ConvNd):
                    idx_layer += 1
                    orig_dim = self.all_dims[idx_layer]
                    deltas = []
                    for idx, dim in enumerate(orig_dim[::-1]):
                        delta_dim = dim - out.size(dim=-(idx + 1))
                        deltas.extend([delta_dim // 2,
                                       delta_dim - delta_dim // 2])
                    out = func.pad(out, deltas)
        else:
            out = out.view(out.size(0), self.output_channels, -1)
        return Normal(loc=out, scale=torch.exp(0.5 * self.w_out_logvar))
