##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from numpy import asanyarray, cov, power
from sklearn.base import BaseEstimator
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_array, check_is_fitted


class KernelMetric(BaseEstimator):
    """Interface for fast weighting matrix computation.
    
    It computes a weighting matrix :math:`W` between input samples based on
    Kernel Density Estimation (KDE) [1]_, [2]_. Concretely, it computes the
    following weighting matrix between multivariate samples
    :math:`x_1, ..., x_n \\in \\mathbb{R}^{d}`:
    
    .. math::
        W_{i,j} = K\\left( H^{-\\frac{1}{2}} (x_i-x_j) \\right)

    with :math:`K` a kernel (or "weighting function") such that:

    - :math:`K(x) \\ge 0` (positive)
    - :math:`\\int K(x) dx = 1` (normalized)
    - :math:`K(x) = K(-x)` (symmetric)

    and :math:`H\\in \\mathbb{R}^{d\\times d}` is the bandwidth in the KDE
    estimation of `p(X)`.

    :math:`H` is a symmetric definite-positive and it can be automatically
    computed based on Scott's rule [3]_ or Silverman's rule [4]_ if required.
    In that case, the bandwidth is computed as a scaled version of the
    diagonal terms in the data covariance matrix:
    
    .. math::
        H \\propto \\mathrm{diag}(\\hat{\\Sigma})

    Parameters
    ----------
    kernel: {'gaussian', 'epanechnikov', 'exponential', 'linear', 'cosine'},\
        default='gaussian'
        The kernel applied to the distance between samples.
    bandwidth: {'scott', 'silverman'} or float or list of float,\
        default="scott"
        The method used to calculate the estimator bandwidth:

        - If `bandwidth` is 'scott' or 'silverman', :math:`H` is a scaled
          version of the diagonal terms in the data covariance matrix.
        - If `bandwidth` is scalar (float or int), :math:`H` is set to a
          diagonal matrix:
          :math:`H = \\mathrm{diag}([bandwidth,\\ldots, bandwidth])`.
        - If `bandwidth` is a list of floats, :math:`H` is a diagonal matrix
          with the list values on the diagonal:
          :math:`H = \\mathrm{diag}(\\text{bandwidth})`.
        - If `bandwidth` is a 2d array, it must be of shape
          `(n_features, n_features)`
    
    Notes
    -----
    Scott's Rule [1]_ estimates the bandwidth as:

    .. math::
        H = \\hat{\\Sigma} \\cdot n^{-\\frac{2}{d+4}}

    where :math:`\\hat{\\Sigma}` is the covariance matrix of the data,
    :math:`n` is the number of samples, and :math:`d` is the number of
    features (:math:`d=1` for univariate data). Here, we only consider
    the diagonal terms (assuming features decorrelation) for numerical
    stability.

    Silverman's rule of thumb [2]_ for multivariate data is:

    .. math::
        H = \\hat{\\Sigma} \\cdot \\left(\\frac{n(d+2)}{4}\\right)^
        {-\\frac{2}{d+4}}

        
    References
    ----------
    .. [1] Rosenblatt, M. (1956). "Remarks on some nonparametric estimates of a
           density function". Annals of Mathematical Statistics.
    .. [2] Parzen, E. (1962). "On estimation of a probability density function
           and mode". Annals of Mathematicals Statistics.
    .. [3] Scott, D. W. (1992). "Multivariate Density Estimation: Theory,
           Practice, and Visualization". Wiley.
    .. [4] Silverman, B. W. (1986). "Density Estimation for Statistics and Data
           Analysis". Monographs on Statistics and Applied Probability.

    """

    def __init__(
        self,
        kernel="gaussian",
        bandwidth: Union[str, float, list[float], np.ndarray] = "scott",
    ):
        self.kernel = self._validate_kernel(kernel)
        self.bandwidth = bandwidth
        self.is_fitted = False

        # Get covariance factor from bandwidth estimator
        if isinstance(self.bandwidth, str):
            if self.bandwidth == "scott":
                self.covariance_factor = self.scotts_factor
            elif self.bandwidth == "silverman":
                self.covariance_factor = self.silverman_factor
        elif isinstance(self.bandwidth, (float, int, list, np.ndarray)):
            pass  # scalar bandwidth, no covariance factor needed
        else:
            raise ValueError(
                "`bandwidth` should be a string ('scott' or 'silverman'), "
                "a scalar (float or int), a list of floats or array got "
                f"{type(self.bandwidth)}"
            )

    def fit(self, X):
        """Computes the bandwidth in the kernel density estimation.

        Parameters
        ----------
        X: array of shape (n_samples, n_features)
            Input data used to estimate the bandwidth (based on covariance
            matrix).

        Returns
        -------
        self: KernelMetric
        """
        X = check_array(self.atleast_2d(X))
        self.n_, self.d_ = X.shape[0], X.shape[1]
        self.set_bandwidth(X)
        return self

    def set_bandwidth(self, X):
        """Compute the estimator bandwidth. Implementation from scipy.

        The new bandwidth calculated after a call to `set_bandwidth` is used
        for subsequent evaluation of the estimated density.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Input data.
        """
        X = check_array(self.atleast_2d(X))
        self.bandwidth = self._check_bandwidth(self.bandwidth, X)

        if isinstance(self.bandwidth, str):
            factor = self.covariance_factor()
            covariance = self.atleast_2d(cov(X, rowvar=False, bias=False))
            # Removes non-diagonal term in covariance matrix to produce
            # bandwidth estimator. Computes square root inverse of covar
            # matrix (can be prone to error...)
            _data_sqr_cov = np.sqrt(np.diag(np.diag(covariance)))
            _data_inv_sqr_cov = np.divide(
                1.0,
                _data_sqr_cov,
                out=np.zeros_like(_data_sqr_cov),
                where=_data_sqr_cov != 0,
            )
            self.sqr_bandwidth_ = _data_sqr_cov * factor
            self.inv_sqr_bandwidth_ = _data_inv_sqr_cov / factor
            print(
                "Square-root bandwidth ('sigma') automatically set to:\n",
                self.sqr_bandwidth_,
            )
        else:
            # simple case: diagonal matrix
            if np.allclose(
                self.bandwidth, np.diag(np.diagonal(self.bandwidth))
            ):
                self.sqr_bandwidth_ = np.sqrt(self.bandwidth)
                self.inv_sqr_bandwidth_ = np.divide(
                    1.0,
                    self.sqr_bandwidth_,
                    out=np.zeros_like(self.sqr_bandwidth_),
                    where=self.sqr_bandwidth_ != 0,
                )
            else:  # harder case
                M = self.bandwidth  # short name
                # Check symmetry
                if not np.allclose(M, M.T, atol=1e-8):
                    raise ValueError("`bandwidth` is not symmetric")
                # Eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(M)

                # Check positive definiteness
                if np.any(eigvals <= 0):
                    raise ValueError("`bandwidth` is not positive definite")

                # Compute sqrt and inverse sqrt of eigenvalues
                sqrt_eigvals = np.sqrt(eigvals)
                invsqrt_eigvals = 1.0 / sqrt_eigvals

                # Reconstruct matrices
                self.sqr_bandwidth_ = (
                    eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T
                )
                self.inv_sqr_bandwidth_ = (
                    eigvecs @ np.diag(invsqrt_eigvals) @ eigvecs.T
                )

    def _check_bandwidth(self, bandwidth, X):
        if isinstance(bandwidth, str):
            return bandwidth
        else:
            if np.isscalar(bandwidth):
                bandwidth = [bandwidth for _ in range(X.shape[1])]
            bandwidth = check_array(bandwidth, ensure_2d=False)
            n_features = None
            if bandwidth.ndim in [1, 2]:
                n_features = bandwidth.shape[0]
                if n_features != X.shape[1]:
                    raise ValueError(
                        "Length of `bandwidth` must match number of features "
                        f"in X, got {n_features}, expected {X.shape[1]}."
                    )
                if bandwidth.ndim == 2 and (n_features != bandwidth.shape[1]):
                    raise ValueError("`bandwidth` must be a squared matrix.")
                if bandwidth.ndim == 1:
                    bandwidth = np.diag(bandwidth)
            else:
                raise ValueError(
                    "`bandwidth` must be 1d or 2d array, got "
                    f"{bandwidth.ndim}d"
                )
            if np.any(bandwidth < 0):
                raise ValueError("`bandwidth` must be positive")
            return bandwidth

    def scotts_factor(self):
        """Compute Scott's factor.

        Returns
        -------
        s: float
            Scott's factor.
        """
        check_is_fitted(self, attributes=["n_", "d_"])
        return power(self.n_, -1.0 / (self.d_ + 4))

    def silverman_factor(self):
        """Compute the Silverman factor.

        Returns
        -------
        s : float
            The silverman factor.
        """
        check_is_fitted(self, attributes=["n_", "d_"])
        return power(self.n_ * (self.d_ + 2.0) / 4.0, -1.0 / (self.d_ + 4))

    def pairwise(self, X):
        """
        Parameters
        ----------
        X: array of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        S: array of shape (n_samples, n_samples)
            Similarity matrix between input data.
        """
        check_is_fitted(self, attributes=["inv_sqr_bandwidth_"])

        X = check_array(self.atleast_2d(X))
        if X.shape[1] != self.d_:
            raise ValueError(
                f"Wrong data dimension, got {X.shape[1]} (expected {self.d_})"
            )
        X_transformed = X @ self.inv_sqr_bandwidth_.T
        pdist = pairwise_distances(X_transformed, metric="euclidean")
        S = self.kernel(pdist)
        return S

    def fit_pairwise(self, X):
        self.fit(X)
        return self.pairwise(X)

    def atleast_2d(self, X):
        X = asanyarray(X)
        if X.ndim == 0:
            X = X.reshape(1, 1)
        elif X.ndim == 1:
            X = X[:, np.newaxis]
        return X

    def _validate_kernel(self, kernel):
        if isinstance(kernel, str) and hasattr(self, kernel.capitalize()):
            return getattr(self, kernel.capitalize())()
        raise NotImplementedError(f"Unknown kernel: {kernel}")

    class Gaussian:
        def __call__(self, x):
            return np.exp(-(x**2) / 2)

    class Epanechnikov:
        def __call__(self, x):
            return (1 - x**2) * (np.abs(x) < 1)

    class Exponential:
        def __call__(self, x):
            return np.exp(-x)

    class Linear:
        def __call__(self, x):
            return (1 - x) * (np.abs(x) < 1)

    class Cosine:
        def __call__(self, x):
            return np.cos(np.pi * x / 2.0) * (np.abs(x) < 1)


class YAwareInfoNCE(nn.Module):
    """
    Implementation of the y-Aware InfoNCE loss [1]_.

    Compute the y-Aware InfoNCE loss, which integrates auxiliary
    information into contrastive learning by weighting sample pairs.

    Given a mini-batch of size :math:`n`, two embeddings
    :math:`z_1=(z_1^i)_{i\in [1..n]}` and :math:`z_2=(z_2^i)_{i\in [1..n]}`
    representing two views of the same samples and a weighting
    matrix :math:`W=(w_{i,j})_{i,j\in [1..n]}` computed using auxiliary
    variables :math:`y`, the loss is:

    .. math::
        \mathcal{L}_{NCE}^y = -\\frac{1}{n} \\sum_{i,j} \\frac{w_{i,j}} \
        {\\sum_{k=1}^{n} w_{i, k}} \\log \\frac{\\exp(\\text{sim}(z_1^{i}, \
        z_2^{j}) / \\tau)}{\\sum_{k=1}^{n} \\exp(\\text{sim}(z_1^{i}, z_2^{k})\
        / \\tau)}

    where :math:`sim` is the cosine similarity,  :math:`\\tau` is the
    temperature and :math:`w_{i,j}` is computed with a kernel :math:`K`
    (e.g. Gaussian) and bandwidth :math:`H` as:

    .. math::
        w_{i,j} = K\\left( H^{-\\frac{1}{2}} (y_i-y_j) \\right)

    
    Parameters
    ----------
    kernel: str in {'gaussian', 'epanechnikov', 'exponential', 'linear', \
        'cosine'}, default='gaussian'
        Kernel to compute the weighting matrix between auxiliary variables.
        See PhD thesis, Dufumier 2022 page 94-95.
    bandwidth: Union[float, int, List[float], array, KernelMetric], default=1.0
        The method used to calculate the bandwidth (:math:`\\sigma^2` in [1]_)
        between auxiliary variables:

        - If `bandwidth` is a scalar (int or float), it sets the bandwidth to
          a diagnonal matrix with equal values.
        - If `bandwidth` is a 1d array, it sets the bandwidth to a
          diagonal matrix and it must be of size equal to the number of
          features in `y`.
        - If bandwidth is a 2d array, it must be of shape
          `(n_features, n_features)` where `n_features` is the number of
          features in `y`.
        - If `bandwidth` is `KernelMetric`, it uses the `pairwise` method to
          compute the similarity matrix between auxiliary variables.
    temperature: float, default=0.1
        Temperature used to scale the dot-product between embedded vectors


    References
    ----------
    .. [1] Dufumier, B., et al., "Contrastive learning with continuous proxy
           meta-data for 3D MRI classification." MICCAI, 2021.
           https://arxiv.org/abs/2106.08808

    """

    def __init__(
        self,
        kernel: str = "gaussian",
        bandwidth: Union[float, list[float], np.ndarray, KernelMetric] = 1.0,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.sim_metric = PairwiseCosineSimilarity()
        if isinstance(bandwidth, KernelMetric):
            self.kernel_metric = bandwidth
        elif isinstance(bandwidth, (int, float, list, np.ndarray)):
            self.kernel_metric = KernelMetric(kernel, bandwidth)
        else:
            raise ValueError(
                "`bandwidth` must be a float, list of float, "
                f"array or KernelMetric, got {type(bandwidth)}"
            )
        self.temperature = temperature
        self.INF = 1e8

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        z1: torch.Tensor of shape (batch_size, n_features)
            First embedded view.
        z2: torch.Tensor of shape (batch_size, n_features)
            Second embedded view.
        labels: Optional[torch.Tensor] of shape (batch_size, n_labels)
            Auxiliary variables associated to the input data.
            If None, the standard InfoNCE loss is returned.

        Returns
        -------
        loss: torch.Tensor
            The y-Aware InfoNCE loss computed between `z1` and `z2`.
        """

        n = len(z1)
        assert len(z1) == len(z2), (
            f"Two tensors z1, z2 must have same shape, got "
            f"{z1.shape} != {z2.shape}"
        )
        if labels is not None:
            assert len(labels) == n, (
                f"Labels length {len(labels)} != vectors length {n}"
            )

        # Computes similarity matrices, shape (N, N)
        sim_z11 = self.sim_metric(z1, z1) / self.temperature
        sim_z22 = self.sim_metric(z2, z2) / self.temperature
        sim_z12 = self.sim_metric(z1, z2) / self.temperature
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_z11 = sim_z11 - self.INF * torch.eye(n, device=z1.device)
        sim_z22 = sim_z22 - self.INF * torch.eye(n, device=z2.device)

        # Stack similarity matrices, shape (2N, 2N)
        sim_z = torch.cat(
            [
                torch.cat([sim_z11, sim_z12], dim=1),
                torch.cat([sim_z12.T, sim_z22], dim=1),
            ],
            dim=0,
        )

        if labels is None:
            correct_pairs = torch.arange(n, device=z1.device).long()
            loss_1 = nn.functional.cross_entropy(
                torch.cat([sim_z12, sim_z11], dim=1), correct_pairs
            )
            loss_2 = nn.functional.cross_entropy(
                torch.cat([sim_z12.T, sim_z22], dim=1), correct_pairs
            )
            loss = (loss_1 + loss_2) / 2.0
        else:
            all_labels = (
                labels.view(n, -1).repeat(2, 1).detach().cpu().numpy()
            )  # [2N, *]

            try:
                check_is_fitted(self.kernel_metric)
            except Exception:
                if isinstance(self.bandwidth, KernelMetric):
                    raise ValueError(
                        "If `bandwidth` is a KernelMetric, it should be "
                        "already fitted on your training data "
                    ) from None
                # Safely fit the kernel to get the correct bandwidth
                self.kernel_metric.fit(all_labels)

            weights = self.kernel_metric.pairwise(all_labels).astype(
                float
            )  # [2N, 2N]
            weights = weights * (
                1 - np.eye(2 * n, dtype=float)
            )  # Put zeros on the diagonal
            weights /= weights.sum(axis=1)
            log_sim_z = nn.functional.log_softmax(sim_z, dim=1)
            loss = (
                -1.0
                / n
                * (torch.from_numpy(weights).to(z1.device) * log_sim_z).sum()
            )
        return loss

    def __str__(self):
        return (
            f"{type(self).__name__}(temp={self.temperature}, "
            f"kernel={self.kernel}, bandwidth={self.bandwidth})"
        )


class PairwiseCosineSimilarity(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x1, x2):
        x1 = nn.functional.normalize(x1, p=2, dim=self.dim)
        x2 = nn.functional.normalize(x2, p=2, dim=self.dim)
        return x1 @ x2.T
