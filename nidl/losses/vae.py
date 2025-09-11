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


class BaseLoss:
    """ Base class for losses.
    """
    def __init__(self, steps_anneal=0, use_mse=False):
        """ Init class.

        Parameters
        ----------
        steps_anneal: int, default 0
            number of annealing steps where gradually adding the
            regularisation.
        use_mse: bool, default False
            if set use MSE for the reconstruction loss rather than Log
            Likelihood.
        """
        self.n_train_steps = 0
        self.layer_outputs = None
        self.steps_anneal = steps_anneal
        self.use_mse = use_mse
        self.cache = {}

    def get_params(self):
        """ Get forward layers outputs.

        Returns
        -------
        q: torch.distributions
            probabilistic encoder (or estimated posterior probability
            function).
        z: torch.Tensor
            the compressed code learned in the bottleneck layer.
        model: nn.Module
            the network.
        """
        if self.layer_outputs is None:
            raise ValueError("The model needs to return the latent space "
                             "distribution parameters q and sampling z as "
                             "well as the model itself.")
        z = self.layer_outputs["z"]
        q = self.layer_outputs["q"]
        model = self.layer_outputs["model"]
        return q, z, model

    def reconstruction_loss(self, p, data):
        """ Calculates the per image reconstruction loss for a batch of data
        (i.e. negative log likelihood).

        The distribution of the likelihood on the each pixel implicitely
        defines the loss. Bernoulli corresponds to a binary cross entropy.
        Gaussian distribution corresponds to MSE, and is sometimes
        used, but hard to train because it ends up focusing only a few
        pixels that are very wrong. Laplace distribution corresponds to
        L1 solves partially the issue of MSE.

        Parameters
        ----------
        p: torch.distributions
            probabilistic decoder (or likelihood of generating true data
            sample given the latent code).
        data: torch.Tensor
            reference data.

        Returns
        -------
        loss: torch.Tensor
            per image cross entropy (i.e. normalized per batch but not pixel
            and channel).
        """
        if isinstance(p, Bernoulli):
            loss = func.binary_cross_entropy(p.probs, data, reduction="sum")
        elif isinstance(p, Normal):
            if self.use_mse:
                loc = p.loc
                if loc.shape != data.shape:
                    loc = torch.squeeze(p.loc)
                loss = func.mse_loss(loc, data, reduction="sum")
            else:
                loss = self.compute_ll(p, data)
                loss = loss.mean(1)
                loss = loss.sum(0)
        elif isinstance(p, Laplace):
            loss = func.l1_loss(p.loc, data, reduction="sum")
            # empirical value to give similar values than bernoulli => use
            # same hyperparam
            loss = loss * 3
            loss = loss * (loss != 0)  # masking to avoid nan
        else:
            raise ValueError(f"Unkown distribution: {p}")

        batch_size = len(data)
        loss = loss / batch_size

        if "iteration" not in self.cache:
            self.cache["iteration"] = 0
        self.cache["iteration"] += len(data)
        self.cache.setdefault("ll", []).append(loss.detach().cpu().numpy())

        return loss

    def compute_ll(self, p, data):
        """ Compute log likelihood.

        Parameters
        ----------
        p: torch.distributions
            probabilistic decoder (or likelihood of generating true data
            sample given the latent code).
        data: torch.Tensor
            reference data.
        """
        loss = - p.log_prob(data)
        shape = loss.shape
        return loss.view(shape[0], shape[1], -1).sum(-1, keepdim=True)

    def kl_normal_loss(self, q):
        """ Calculates the KL divergence between a normal distribution
        with diagonal covariance and a unit normal distribution.

        Parameters
        ----------
        q: torch.distributions
            probabilistic encoder (or estimated posterior probability
            function).
        """
        dimension_wise_kl = kl_divergence(q, Normal(0, 1)).mean(dim=0)
        self.cache.setdefault("kl", []).append(
            dimension_wise_kl.detach().cpu().numpy())
        return dimension_wise_kl.sum()

    def kl_log_uniform(self, normal):
        """ Calculates the KL log uniform divergence.

        Paragraph 4.2 from:
        Variational Dropout Sparsifies Deep Neural Networks
        Molchanov, Dmitry; Ashukha, Arsenii; Vetrov, Dmitry
        https://arxiv.org/abs/1701.05369
        https://github.com/senya-ashukha/variational-dropout-sparsifies-dnn/
        blob/master/KL%20approximation.ipynb
        """
        mu = normal.loc
        logvar = normal.scale.pow(2).log()
        log_alpha = BaseLoss.compute_log_alpha(mu, logvar)
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        neg_kl = (k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 *
                  torch.log1p(torch.exp(-log_alpha)) - k1)
        neg_kl = neg_kl.mean(dim=0)
        self.cache.setdefault("kl", []).append(neg_kl.detach().cpu().numpy())
        return - neg_kl.sum()

    @staticmethod
    def compute_log_alpha(mu, logvar):
        return (logvar - 2 * torch.log(torch.abs(mu) + 1e-8)).clamp(
            min=-8, max=8)

    def linear_annealing(self, init, fin):
        """ Linear annealing of a parameter.

        Returns
        -------
        annealed: float
            loss factor to gradually add the regularisation.
        """
        if self.steps_anneal == 0:
            return fin
        assert fin > init
        delta = fin - init
        annealed = min(init + delta * self.n_train_steps /
                       self.steps_anneal, fin)
        return annealed

    def update_train_step(self, iteration=None):
        """ Update the train step.

        Parameters
        ----------
        iteration: int, default None
            the current iteration.
        """
        if iteration is None and "iteration" in self.cache:
            iteration = self.cache["iteration"]
        if iteration is None:
            raise ValueError("No iteration specified.")
        self.n_train_steps = iteration


class BetaHLoss(BaseLoss):
    """ Compute the Beta-VAE loss.

    beta-VAE: Learning Basic Visual Concepts with a Constrained Variational
    Framework, Irina Higgins, ICLR 2017.
    """
    def __init__(self, beta=4, **kwargs):
        """ Init class.

        Parameters
        ----------
        beta: float, default 4
            weight of the kl divergence.
        kwargs: dict
            additional arguments for 'BaseLoss'.
        """
        super().__init__(**kwargs)
        self.beta = beta

    def __call__(self, p, data, **kwargs):
        """ Compute the loss.
        """
        q, z, model = self.get_params()
        rec_loss = self.reconstruction_loss(p, data)
        kl_loss = self.kl_normal_loss(q)
        if model.training:
            anneal_reg = self.linear_annealing(init=0, fin=1)
            self.update_train_step()
        else:
            anneal_reg = 1
        kl_loss = anneal_reg * (self.beta * kl_loss)
        loss = rec_loss + kl_loss
        self.cache.setdefault("loss", []).append(loss.detach().cpu().numpy())
        return loss, {"rec_loss": rec_loss, "kl_loss": kl_loss}


class BetaBLoss(BaseLoss):
    """ Compute the Beta-VAE loss.

    Understanding disentangling in beta-VAE, Burgess, arXiv 2018.
    """
    def __init__(self, C_init=0., C_fin=20., gamma=100., **kwargs):
        """ Init class.

        Parameters
        ----------
        C_init: float, default 0
            starting annealed capacity C.
        C_fin: float, default 20
            final annealed capacity C.
        gamma: float, default 100
            weight of the KL divergence term.
        kwargs: dict
            additional arguments for 'BaseLoss'.
        """
        super().__init__(**kwargs)
        self.gamma = gamma
        self.C_init = C_init
        self.C_fin = C_fin

    def __call__(self, p, data, **kwargs):
        """ Compute the loss.
        """
        q, z, model = self.get_params()
        rec_loss = self.reconstruction_loss(p, data)
        kl_loss = self.kl_normal_loss(q)
        if model.training:
            C = self.linear_annealing(init=self.C_init, fin=self.C_fin)
            self.update_train_step()
        else:
            C = self.C_fin
        kl_loss = self.gamma * (kl_loss - C).abs()
        loss = rec_loss + kl_loss
        return loss, {"rec_loss": rec_loss, "kl_loss": kl_loss}


class SparseLoss(BaseLoss):
    """ Compute the Beta-Sparse VAE loss.

    Sparse Multi-Channel Variational Autoencoder for the Joint Analysis of
    Heterogeneous Data, Luigi Antelmi, Nicholas Ayache, Philippe Robert,
    Marco Lorenzi, PMLR 2019.
    """
    def __init__(self, beta=4, **kwargs):
        """ Init class.

        Parameters
        ----------
        beta: float, default 4
            weight of the kl divergence.
        kwargs: dict
            additional arguments for 'BaseLoss'.
        """
        super().__init__(**kwargs)
        self.beta = beta

    def __call__(self, p, data, **kwargs):
        """ Compute the loss.
        """
        q, z, model = self.get_params()
        rec_loss = self.reconstruction_loss(p, data)
        kl_loss = self.kl_log_uniform(q)
        if model.training:
            anneal_reg = self.linear_annealing(init=0, fin=1)
        else:
            anneal_reg = 1
        kl_loss = anneal_reg * (self.beta * kl_loss)
        loss = rec_loss + kl_loss
        return loss, {"rec_loss": rec_loss, "kl_loss": kl_loss}


class BtcvaeLoss(BaseLoss):
    """ Compute the decomposed KL loss with either minibatch weighted
    sampling or minibatch stratified sampling according.

    Isolating sources of disentanglement in variational autoencoders, Tian Qi,
    Advances in Neural Information Processing Systems, 2018.
    """

    def __init__(self, dataset_size, alpha=1., beta=6., gamma=1.,
                 is_mss=True, **kwargs):
        """ Init class.

        Parameters
        ----------
        dataset_size: int
            number of training images in the dataset.
        alpha: float, default 1
            weight of the mutual information term.
        beta: float, default 6
            weight of the total correlation term.
        gamma: float, default 1
            weight of the dimension-wise KL term.
        is_mss: bool, default True
            wether to use minibatch stratified sampling instead of minibatch
            weighted sampling.
        kwargs: dict
            additional arguments for 'BaseLoss'.
        """
        super().__init__(**kwargs)
        self.dataset_size = dataset_size
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.is_mss = is_mss

    def __call__(self, p, data):
        """ Compute the loss.
        """
        q, z, model = self.get_params()
        rec_loss = self.reconstruction_loss(p, data)
        log_pz, log_qz, log_prod_qzi, log_qz_x = self.get_probs(z, q)
        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss = (log_qz_x - log_qz).mean()
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qz - log_prod_qzi).mean()
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        dw_kl_loss = (log_prod_qzi - log_pz).mean()
        if model.training:
            anneal_reg = self.linear_annealing(init=0, fin=1)
            self.update_train_step()
        else:
            anneal_reg = 1
        mi_loss = self.alpha * mi_loss
        tc_loss = self.beta * tc_loss
        dw_kl_loss = anneal_reg * self.gamma * dw_kl_loss
        loss = rec_loss + mi_loss + tc_loss + dw_kl_loss
        return loss, {"mi_loss": mi_loss, "tc_loss": tc_loss,
                      "dw_kl_loss": dw_kl_loss}

    def get_probs(self, z, q):
        # Calculate log q(z|x)
        log_qz_x = q.log_prob(z).sum(dim=1)

        # Calculate log p(z)
        pz = Normal(loc=torch.zeros_like(z), scale=1)
        log_pz = pz.log_prob(z).sum(dim=1)

        # Calculate log q(z)
        batch_size = len(z)
        mat_log_qz = BtcvaeLoss.matrix_log_density_gaussian(z, q)
        if self.is_mss:
            log_iw_mat = BtcvaeLoss.log_importance_weight_matrix(
                batch_size, self.dataset_size)
            log_iw_mat = torch.unsqueeze(log_iw_mat, dim=-1).to(z.device)
            mat_log_qz = mat_log_qz + log_iw_mat
        log_qz = torch.logsumexp(mat_log_qz.sum(dim=2), dim=1, keepdim=False)
        log_prod_qzi = torch.logsumexp(
            mat_log_qz, dim=1, keepdim=False).sum(dim=1)

        return log_pz, log_qz, log_prod_qzi, log_qz_x

    @staticmethod
    def matrix_log_density_gaussian(x, q):
        """ Calculates log density of a Gaussian for all combination of bacth
        pairs of 'x' and 'mu', i.e. return tensor of shape (batch_size,
        batch_size, dim) instead of (batch_size, dim) in the usual log density.

        Parameters
        ----------
        x: torch.Tensor (batch_size, dim)
            value at which to compute the density.
        q: torch.distributions
            probabilistic encoder (or estimated posterior probability
            function).
        """
        x = torch.unsqueeze(x, dim=1)
        _mu = torch.unsqueeze(q.loc, dim=0)
        _sigma = torch.unsqueeze(q.scale, dim=0)
        _q = Normal(loc=_mu, scale=_sigma)
        return _q.log_prob(x)

    @staticmethod
    def log_importance_weight_matrix(batch_size, dataset_size):
        """ Calculates a log importance weight matrix.

        Parameters
        ----------
        batch_size: int
            number of training images in the batch.
        dataset_size: int
            number of training images in the dataset.
        """
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M + 1] = 1 / N
        W.view(-1)[1::M + 1] = strat_weight
        W[M - 1, 0] = strat_weight
        return W.log()
