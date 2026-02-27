##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import numpy as np
import torch

def alignment_score(
    z1, z2, normalize: bool = True, alpha: int = 2, eps: float = 1e-12
):
    """
    Compute the alignment score between two embeddings [1]_.

    This metric measures how closely aligned two embeddings :math:`z1` and
    :math:`z2`. It corresponds to the expected powered Euclidean distance
    between embeddings. Lower values = better alignment.

    Formally:

    .. math::

        \\text{Alignment}(z_1, z_2)
        = \\frac{1}{n}\\sum_{i=1}^n
        \\lVert z_1^{(i)} - z_2^{(i)} \\rVert_2^{\\alpha}

    with :math:`z_1=(z_1^{(1)}, ..., z_1^{(n)})` and
    :math:`z_2=(z_2^{(1)}, ..., z_2^{(n)})`

    Parameters
    ----------
    z1: torch.Tensor or np.ndarray, shape (n_samples, n_features)
        Embeddings from the first view / augmentation.

    z2: torch.Tensor or np.ndarray, shape (n_samples, n_features)
        Embeddings from the second view / augmentation.
        Must have the same shape as `z1`.

    normalize: bool, default=True
        If True, each vector is L2-normalized before computing the alignment,
        as done in contrastive methods that operate on the unit hypersphere
        (SimCLR, MoCo, etc.).

    alpha: int or float, default=2
        Exponent applied to the Euclidean distance.
        - `alpha=2` corresponds to the original definition in the paper.
        - `alpha=1` gives average L2 distance.

    eps: float, default=1e-12
        Small value added to avoid division by zero.

    Returns
    -------
    score : torch.Tensor or numpy scalar
        The alignment score. Lower is better.

        - If inputs are tensors → returns a 0-dim `torch.Tensor`.
        - If inputs are NumPy arrays → returns a `numpy.float64`.

    References
    ----------
    .. [1] T. Wang, P. Isola, "Understanding Contrastive Representation
           Learning through Alignment and Uniformity on the Hypersphere",
           ICML 2020.
    """

    if isinstance(z1, np.ndarray) and isinstance(z2, np.ndarray):
        if normalize:
            z1 = z1 / (np.linalg.norm(z1, axis=1, keepdims=True) + eps)
            z2 = z2 / (np.linalg.norm(z2, axis=1, keepdims=True) + eps)
        diff = z1 - z2
        dist = np.linalg.norm(diff, axis=1) ** alpha
        return dist.mean()

    if isinstance(z1, torch.Tensor) and isinstance(z2, torch.Tensor):
        if normalize:
            z1 = torch.nn.functional.normalize(z1, dim=1, eps=eps)
            z2 = torch.nn.functional.normalize(z2, dim=1, eps=eps)
        diff = z1 - z2
        dist = diff.norm(dim=1).pow(alpha)
        return dist.mean()

    raise TypeError(
        f"`z1` and `z2` must be either torch.Tensor or np.ndarray, "
        f"got ({type(z1), type(z2)})"
    )


def uniformity_score(z, normalize: bool = True, t: float = 2.0, eps=1e-12):
    """
    Compute the uniformity score of an embedding [1]_

    This metric measures how uniform the embedding vectors are distributed
    on the unit hypersphere. Lower values = more uniform distribution.

    It is defined as the log of the expected Gaussian
    potential over all non-identical pairs:

    .. math::

        U(z) = \\log \\frac{1}{n(n-1)}\\sum_{i \\ne j}
               \\exp\\left(-t \\, \\lVert z_i - z_j \\rVert_2^2 \\right)

    where all vectors are first normalized to lie on the unit hypersphere.

    Lower uniformity values  = more uniform distribution, which is generally
    considered better for contrastive representation learning.

    Parameters
    ----------
    z: torch.Tensor or np.ndarray, shape (n_samples, n_features)
        The embedding vectors.

        - If a NumPy array is provided, computation is performed in NumPy.
        - If a torch.Tensor is provided, computation is performed in PyTorch
          and the returned value is a `torch.Tensor` scalar.

    normalize: bool, default=True
        If True, each vector is L2-normalized before computing the uniformity,
        as done in contrastive methods that operate on the unit hypersphere
        (SimCLR, MoCo, etc.).

    t: float, default=2.0
        Temperature parameter controlling the sharpness of the Gaussian kernel.
        `t = 2` corresponds to the original definition in [1]_.

    eps: float, default=1e-12
        Small value added to avoid division by zero.

    Returns
    -------
    score : torch.Tensor or numpy scalar
        The uniformity score.

        - PyTorch input → returns a 0-dim `torch.Tensor`
        - NumPy input  → returns a `numpy.float64`

    References
    ----------
    .. [1] T. Wang, P. Isola, "Understanding Contrastive Representation
           Learning through Alignment and Uniformity on the Hypersphere",
           ICML 2020.
    """

    if isinstance(z, np.ndarray):
        if normalize:
            # Normalize to unit hypersphere
            z = z / (np.linalg.norm(z, axis=1, keepdims=True) + eps)

        # Compute pairwise squared Euclidean distances
        # Efficient formulation: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
        sim = z @ z.T  # cosine similarities (since normalized)
        dist_sq = 2 - 2 * sim

        # Exclude diagonal i == j
        mask = ~np.eye(len(z), dtype=bool)
        dist_sq = dist_sq[mask]

        # Uniformity score
        return np.log(np.exp(-t * dist_sq).mean())

    if isinstance(z, torch.Tensor):
        if normalize:
            z = torch.nn.functional.normalize(z, dim=1, eps=eps)

        # Compute pairwise distances via pdist
        pdist = torch.nn.functional.pdist(z, p=2)  # shape: [n*(n-1)/2]
        score = torch.exp(-t * pdist.pow(2)).mean().log()
        return score

    raise TypeError(
        f"`z` must be either a torch.Tensor or a numpy.ndarray, got {type(z)}"
    )


def contrastive_accuracy_score(
    z1, z2, normalize: bool = True, topk: int = 1, eps: float = 1e-12
):
    """
    Compute the top-k contrastive accuracy between two embeddings.

    This metric measures how often the true positive pair is among the
    top-k most similar candidates in the opposite view, in both directions:

    - For each i, treat ``z1[i]`` as a query and all rows of ``z2`` as a
      retrieval database. Check whether the matching element ``z2[i]`` is
      within the top-k most similar vectors to ``z1[i]``.
    - Symmetrically, treat ``z2[i]`` as a query and all rows of ``z1`` as
      the database, and check whether ``z1[i]`` is within the top-k neighbors.

    The final score is the average of the two directional accuracies:

    .. math::

        \\text{Acc}_{k}(z_1, z_2)
        = \\tfrac{1}{2} \\left(
            \\text{Acc}_{k}(z_1 \\to z_2)
          + \\text{Acc}_{k}(z_2 \\to z_1)
        \\right),

    where each directional accuracy is the fraction of queries whose true
    pair is in the top-k most similar candidates.

    Similarities are Euclidean dot-product between the embeddings. The score is
    in the range ``[0, 1]``, where higher is better.


    Parameters
    ----------
    z1 : torch.Tensor or np.ndarray, shape (n_samples, n_features)
        Embeddings from the first view / augmentation.

    z2 : torch.Tensor or np.ndarray, shape (n_samples, n_features)
        Embeddings from the second view / augmentation.
        Must have the same shape as ``z1``.

    normalize : bool, default=True
        If True, each embedding vector is L2-normalized along the feature
        dimension before computing similarities. This makes the metric
        equivalent to using cosine similarity. If False, raw dot products
        are used.

    topk : int, default=1
        The "k" in "top-k". For each query, we check whether the true
        counterpart index ``i`` is contained in the indices of the top-k
        most similar candidates. If ``topk`` is greater than the number of
        samples, it is automatically clipped.

    eps : float, default=1e-12
        Small constant used to avoid division by zero during normalization.

    Returns
    -------
    score : torch.Tensor or numpy scalar
        The contrastive top-k accuracy:

        - If inputs are ``torch.Tensor`` → returns a 0-dim ``torch.Tensor``.
        - If inputs are ``np.ndarray`` → returns a NumPy scalar.

    Raises
    ------
    TypeError
        If ``z1`` and ``z2`` are not both torch tensors or both NumPy arrays.

    ValueError
        If shapes of ``z1`` and ``z2`` do not match, or if they are not
        2-dimensional, or if ``topk < 1``.

    Examples
    --------
    >>> z1 = torch.randn(8, 128)
    >>> z2 = z1 + 0.1 * torch.randn(8, 128)  # slightly perturbed positives
    >>> contrastive_accuracy_score(z1, z2, topk=1)
    tensor(1.)  # often close to 1 for this synthetic example
    """

    if topk < 1:
        raise ValueError(f"`topk` must be >= 1, got {topk}")

    if isinstance(z1, torch.Tensor) and isinstance(z2, torch.Tensor):
        backend = "torch"
    elif isinstance(z1, np.ndarray) and isinstance(z2, np.ndarray):
        backend = "numpy"
    else:
        raise TypeError(
            "`z1` and `z2` must both be torch.Tensors or both be np.ndarrays, "
            f"got {type(z1)} and {type(z2)}"
        )

    if z1.shape != z2.shape:
        raise ValueError(
            f"`z1` and `z2` must have the same shape, got {z1.shape} and "
            f"{z2.shape}"
        )
    if z1.ndim != 2:
        raise ValueError(
            f"`z1` and `z2` must be 2D (n_samples, n_features), got "
            f"ndim={z1.ndim}"
        )

    n_samples = z1.shape[0]
    if n_samples == 0:
        raise ValueError("Empty embeddings: n_samples == 0")

    k = min(int(topk), n_samples)

    if backend == "numpy":
        z1 = z1.astype(np.float32)
        z2 = z2.astype(np.float32)

        if normalize:
            z1 = z1 / (np.linalg.norm(z1, axis=1, keepdims=True) + eps)
            z2 = z2 / (np.linalg.norm(z2, axis=1, keepdims=True) + eps)

        # Similarity matrix: (n_samples, n_samples)
        sim = z1 @ z2.T  # cosine or dot-product similarities

        # -------- z1 -> z2 direction --------
        # For each row i, get indices of top-k highest values
        topk_idx_12 = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        # Check if true index i is within top-k
        rows = np.arange(n_samples)[:, None]
        hits_12 = (topk_idx_12 == rows).any(axis=1)  # shape: (n_samples,)
        acc_12 = hits_12.mean()

        # -------- z2 -> z1 direction --------
        sim_T = sim.T
        topk_idx_21 = np.argpartition(-sim_T, kth=k - 1, axis=1)[:, :k]
        hits_21 = (topk_idx_21 == rows).any(axis=1)
        acc_21 = hits_21.mean()
        return 0.5 * (acc_12 + acc_21)
    else:
        # Cast to float32 (or at least ensure floating type)
        if not torch.is_floating_point(z1):
            z1 = z1.float()
        if not torch.is_floating_point(z2):
            z2 = z2.float()

        if normalize:
            z1 = torch.nn.functional.normalize(z1, dim=1, eps=eps)
            z2 = torch.nn.functional.normalize(z2, dim=1, eps=eps)

        # Similarity matrix: (n_samples, n_samples)
        sim = torch.matmul(z1, z2.T)

        # -------- z1 -> z2 direction --------
        # topk over last dimension (over all candidates in z2)
        _, idx_12 = torch.topk(sim, k=k, dim=1)  # idx_12: (n_samples, k)
        true_idx = torch.arange(n_samples, device=idx_12.device).unsqueeze(1)
        hits_12 = (idx_12 == true_idx).any(dim=1).float()  # (n_samples,)
        acc_12 = hits_12.mean()

        # -------- z2 -> z1 direction --------
        sim_T = sim.T
        _, idx_21 = torch.topk(sim_T, k=k, dim=1)
        hits_21 = (idx_21 == true_idx).any(dim=1).float()
        acc_21 = hits_21.mean()
        return 0.5 * (acc_12 + acc_21)

def procrustes_similarity(X, Y) -> float:
    """
    Procrustes similarity between two point clouds / embeddings in the Euclidean case, with global scale invariance.
    Computes the least squares solution for the regression problem Y = sΩX + b where Ω is orthogonal, b is a constant vector and s a global scaling parameter.
    Returns the Frobenius cosine between Ŷ = sXΩ+b and Y, i.e. the quantity ⟨Y,Ŷ⟩/‖Y‖⋅‖Ŷ‖ where ⟨⋅,⋅⟩ is the Frobenius inner product of matrices and ‖⋅‖ the associated Frobenius norm.

    https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    Parameters
    ----------
    X : torch.Tensor or array-like, shape (n, d1)
    Y : torch.Tensor or array-like, shape (n, d2)

    Returns
    -------
    sim : 0-dim torch.Tensor or np.ndarray
        In [0, 1]. 1 means that X and Y are identical up to translation + orthogonal transform + global scaling.
    """

    # Type handling
    return_type = "torch"
    if not isinstance(X, torch.Tensor):
        try:
            X = torch.as_tensor(X)
            return_type = "numpy"
        except:
            raise TypeError(f"X must be numeric and array-like")
    if not isinstance(Y, torch.Tensor):
        try:
            Y = torch.as_tensor(Y)
            return_type = "numpy"
        except:
            raise TypeError(f"Y must be numeric and array-like")
                        
    # Mutualize dtype and device
    Y = Y.to(X.device, dtype=X.dtype)

    # Shape checks
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have same number of rows. Got {X.shape[0]} and {Y.shape[0]}.")

    # Center
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)

    # Frobenius norms (scale)
    nx = torch.linalg.norm(Xc, ord="fro")
    ny = torch.linalg.norm(Yc, ord="fro")

    # Degenerate cases: if either cloud is constant after centering, similarity undefined so return 0.0
    if torch.isclose(nx * ny, torch.Tensor([0.]).to(dtype=X.dtype, device=X.device)):
        sim = torch.Tensor(0.0).to(dtype=X.dtype, device=X.device)
    else:
        # Cross-covariance-like matrix
        M = Xc.T @ Yc # shape (d1, d2)

        # Nuclear norm = sum of singular values
        _, svals, _ = torch.linalg.svd(M)
        nuc = svals.sum()

        sim = nuc / (nx * ny)
        
        # Clamp to [0,1] for numerical safety
        sim = torch.clamp(sim, torch.tensor(0).to(dtype=sim.dtype, device=sim.device), torch.tensor(1).to(dtype=sim.dtype, device=sim.device))

    if return_type == "numpy":
        return sim.numpy()
    
    return sim

def procrustes_r2(X, Y) -> float:
    """
    Procrustes similarity between two point clouds / embeddings in the Euclidean case, when scale does not matter,
    i.e. points are considered on the unit hypersphere.
    Computes the least squares solution for the regression problem Y = ΩX + b where Ω is orthogonal, b is a constant vector.
    Returns the (variance-weighted) r^2 of the best fit.

    Note: procrustes_r2(X, Y) != procrustes_r2(Y, X) in general.

    https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    Parameters
    ----------
    X : torch.Tensor or array-like, shape (n, d1)
    Y : torch.Tensor or array-like, shape (n, d2)

    Returns
    -------
    sim : 0-dim torch.Tensor or np.ndarray
        In (-∞, 1]. 1 means that X and Y are identical up to translation + orthogonal transform.
    """

    # Type handling
    return_type = "torch"
    if not isinstance(X, torch.Tensor):
        try:
            X = torch.as_tensor(X)
            return_type = "numpy"
        except:
            raise TypeError(f"X must be numeric and array-like")
    if not isinstance(Y, torch.Tensor):
        try:
            Y = torch.as_tensor(Y)
            return_type = "numpy"
        except:
            raise TypeError(f"Y must be numeric and array-like")
                        
    # Mutualize dtype and device
    Y = Y.to(X.device, dtype=X.dtype)

    # Shape checks
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have same number of rows. Got {X.shape[0]} and {Y.shape[0]}.")

    # Center
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)

    # Degenerate case
    if torch.isclose(torch.norm(Yc, p="fro"), torch.tensor(0.0, dtype=X.dtype)):
        sim = torch.Tensor(0.0).to(dtype=X.dtype, device=X.device)
    else:
        # SVD-based orthogonal Procrustes
        M = Xc.T @ Yc
        U, _, Vh = torch.linalg.svd(M, full_matrices=False)
        R = U @ Vh

        Yhat = Xc @ R

        # Variance-weighted R^2
        ss_res = torch.sum(torch.linalg.norm(Yc - Yhat, dim=1))
        ss_tot = torch.sum(torch.linalg.norm(Yc, dim=1))

        sim = 1.0 - ss_res / ss_tot

    if return_type == "numpy":
        return sim.numpy()
    
    return sim

def kruskal_similarity(X, Y, spherical = False) -> float:
    """
    Kruskal similarity between two point clouds / embeddings.
    Computes the pairwise Euclidean distances or cosine similarities of both point clouds.
    Returns the r^2 score of the upper triangular portion of both matrices.
    
    This essentially a scaled version of Kruskal's stress, generalized to account for either Euclidean distance or cosine similarity
    https://en.wikipedia.org/wiki/Multidimensional_scaling#Non-metric_multidimensional_scaling_(NMDS)

    Note: kruskal_similarity(X, Y) != kruskal_similarity(Y, X) in general.

    Parameters
    ----------
    X : torch.Tensor or array-like, shape (n, d1)
    Y : torch.Tensor or array-like, shape (n, d2)
    
    spherical : bool
        If False, compare Euclidean distances
        If True, compare cosine similarities

    Returns
    -------
    sim : 0-dim torch.Tensor or np.ndarray
        In (-∞, 1]. 1 means X and Y are identical (after normalization if spherical == True) up to Euclidean isometry.
    """

    # Type handling
    return_type = "torch"
    if not isinstance(X, torch.Tensor):
        try:
            X = torch.as_tensor(X)
            return_type = "numpy"
        except:
            raise TypeError(f"X must be numeric and array-like")
    if not isinstance(Y, torch.Tensor):
        try:
            Y = torch.as_tensor(Y)
            return_type = "numpy"
        except:
            raise TypeError(f"Y must be numeric and array-like")
                        
    # Mutualize dtype and device
    Y = Y.to(X.device, dtype=X.dtype)

    # --- Shape checks ---------------------------------------------------------
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have same number of rows. Got {X.shape[0]} and {Y.shape[0]}."
        )
    
    N = X.shape[0]

    if not spherical:    
        D_X = torch.cdist(X, X)
        D_Y = torch.cdist(Y, Y)
    else:
        Xn = torch.nn.functional.normalize(X)
        Yn = torch.nn.functional.normalize(Y)

        D_X = Xn @ Xn.T
        D_Y = Yn @ Yn.T

    idxs_1, idxs_2 = torch.triu_indices(N, N, offset=1)

    D_X = D_X[idxs_1, idxs_2]
    D_Y = D_Y[idxs_1, idxs_2]

    ss_res = torch.sum(torch.linalg.norm(D_X - D_Y))
    ss_tot = torch.sum(torch.linalg.norm(D_Y - torch.mean(D_Y)))

    sim = 1 - ss_res/ss_tot

    if return_type == "numpy":
        return sim.numpy()
    
    return sim