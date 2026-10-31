##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import torch
import torch.nn.functional as func
from torch import nn


class CoMMLoss(nn.Module):
    r"""Normalized Temperature Cross-Entropy loss for multimodal contrastive
    learning as defined in CoMM [1]_, with missing modalities.

    Given the embeddings of each of the :math:`M` modalities, :math:`z^{(m)}`,
    and of all the modalities, :math:`z`, for two augmented views, the loss
    averages the InfoNCE losses [2]_ :math:`\ell` between each embedding of
    one view and the multimodal embedding of the other view:

    .. math::
        \mathcal{L} = \frac{1}{M + 1} \sum_{k=1}^{M + 1}
        \frac{1}{2} \Big(\ell\big(z_1^{(k)}, z_2\big)
        + \ell\big(z_2^{(k)}, z_1\big)\Big)

    with :math:`z^{(M + 1)} = z`. The term of a modality only uses the
    samples having it, and terms with less than two samples are skipped.

    Adapted from https://github.com/Duplums/CoMM.

    Parameters
    ----------
    temperature : float, default=0.1
        Scale logits by the inverse of the temperature.
    weights : sequence of float or None, default=None
        Optional weight of each term (one per modality, then the multimodal
        term).

    References
    ----------
    .. [1] Benoit Dufumier, Javiera Castillo-Navarro, et al., "What to align
           in multimodal contrastive learning?", ICLR 2025.
    .. [2] Aaron van den Oord, Yazhe Li, Oriol Vinyals, "Representation
           Learning with Contrastive Predictive Coding", arXiv 2018.
    """

    def __init__(self, temperature=0.1, weights=None):
        super().__init__()
        if temperature <= 0:
            raise ValueError(
                f"temperature should be positive (got {temperature})")
        self.temperature = temperature
        self.weights = weights
        self.INF = 1e8

    def infonce(self, z1, z2):
        """InfoNCE loss and accuracy between two batches of normalized
        embeddings, where `z1[i]` and `z2[i]` are positive pairs.
        """
        N = len(z1)
        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zii = (z1 @ z1.T) / self.temperature
        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z2 @ z2.T) / self.temperature
        # dim [N, N] => the diag contains the correct pairs (i,j)
        sim_zij = (z1 @ z2.T) / self.temperature
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z1.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z1.device)
        sim_Z = torch.cat([
            torch.cat([sim_zij, sim_zii], dim=1),
            torch.cat([sim_zjj, sim_zij.T], dim=1)], dim=0)
        log_sim_Z = func.log_softmax(sim_Z, dim=1)
        loss = - torch.diag(log_sim_Z).mean()
        # compute SSL accuracy
        with torch.no_grad():
            pred = torch.argmax(sim_zij, dim=1)
            correct = pred.eq(torch.arange(N, device=z1.device)).sum()
            acc = 100 * correct / N
        return loss, acc

    def forward(self, z1, z2, present=None):
        """Forward implementation.

        Parameters
        ----------
        z1 : list of torch.Tensor
            The M + 1 embeddings of the first view, each of shape
            (batch_size, n_features): one per modality, then the multimodal
            one.
        z2 : list of torch.Tensor
            The M + 1 embeddings of the second view.
        present : torch.Tensor or None, default=None
            Bool mask of shape (batch_size, M), True where a sample has a
            modality. If None, all the modalities are present.

        Returns
        -------
        outputs : dict
            The CoMM loss ("loss"), the mean SSL accuracy ("ssl_acc"), and the
            loss and accuracy of each computed term ("ssl_loss_i",
            "ssl_acc_i").
        """
        assert len(z1) == len(z2)
        n_emb = len(z1)
        prototype = n_emb - 1           # the last one is the multimodal one
        n_mod = n_emb - 1
        batch_size = len(z1[prototype])
        if self.weights is not None and len(self.weights) != n_emb:
            raise ValueError(
                f"Expected {n_emb} weights (one per modality, then the "
                f"multimodal term), got {len(self.weights)}.")
        if present is not None and tuple(present.shape) != (batch_size, n_mod):
            raise ValueError(
                f"present must have shape ({batch_size}, {n_mod}), got "
                f"{tuple(present.shape)}.")
        # list of size M+1 with each fused embedding of shape (B,D)
        z1 = [func.normalize(z, p=2, dim=-1) for z in z1]
        z2 = [func.normalize(z, p=2, dim=-1) for z in z2]

        # Apply InfoNCE between the "prototype embedding" and all the others,
        # over the subjects for which each subset is defined.
        loss = []
        acc = []
        kept = []
        for i in range(n_emb):
            if i < n_mod and present is not None:
                valid = present[:, i]           # subjects having modality i
            else:
                valid = torch.ones(z1[i].shape[0], dtype=torch.bool,
                                   device=z1[i].device)     # all subjects
            if int(valid.sum()) < 2:            # InfoNCE needs >= 2 samples
                continue
            loss1, acc1 = self.infonce(z1[i][valid], z2[prototype][valid])
            loss2, acc2 = self.infonce(z2[i][valid], z1[prototype][valid])
            loss.append((loss1 + loss2) / 2.)
            acc.append((acc1 + acc2) / 2.)
            kept.append(i)
        if len(kept) == 0:
            raise ValueError(
                f"CoMMLoss needs at least 2 samples, got {batch_size}.")

        ssl_acc = {f"ssl_acc_{i}": acc_ for i, acc_ in zip(kept, acc)}
        losses = {f"ssl_loss_{i}": loss_ for i, loss_ in zip(kept, loss)}
        if self.weights is not None:
            weights = torch.tensor([self.weights[i] for i in kept],
                                   device=z1[0].device)
            loss = torch.mean(torch.stack(loss) * weights)
        else:
            loss = torch.mean(torch.stack(loss))
        acc = torch.mean(torch.stack(acc))
        return {"loss": loss, "ssl_acc": acc, **ssl_acc, **losses}

    def __str__(self):
        return f"{type(self).__name__}(temp={self.temperature})"
