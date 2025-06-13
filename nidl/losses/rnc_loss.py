import torch
import torch.nn as nn

""" RnC loss [1] adapted from https://github.com/kaiwenzha/Rank-N-Contrast 

[1] Rank-N-Contrast: Learning Continuous Representations for Regression, Zha et al., NeurIPS 2023
"""

class LabelDifference(nn.Module):
    def __init__(self, distance_type: str='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels: torch.Tensor):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        diffs = labels[:, None, :] - labels[None, :, :]
        if self.distance_type == 'l1':
            return torch.abs(diffs).sum(dim=-1)
        elif self.distance_type == 'l2':
            return diffs.norm(2, dim=-1)       
        else:
            raise ValueError(self.distance_type)


class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type: str='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features: torch.Tensor):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        else:
            raise ValueError(self.similarity_type)


class RnCLoss(nn.Module):
    def __init__(self, 
                 temperature: float=2.0, 
                 label_diff: str='l1', 
                 feature_sim: str='l2'):
        """
        Parameters
        ----------
        temperature: float, default=2.0
            Scaling parameter in the similarity function between two latent representations.
        
        label_diff: str in {'l1', 'l2'}, default='l1'
            Which distance to use between labels, ultimately used to rank 
            the samples according to this distance matrix.
        
        feature_sim: str in {'l2'}, default='l2'
            Which similarity metric to use between feature embeddings. 
            Currently, only negative L2 is implemented.
        """

        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, z1, z2, labels):
        """
        Parameters
        ----------
        z1: torch.Tensor of shape (batch_size, n_features)
            First embedded view.

        z2: torch.Tensor of shape (batch_size, n_features) or None
            Second embedded view.

        labels: torch.Tensor of shape (batch_size, n_labels)
            The corresponding labels. 

        Returns
        ----------
        loss: torch.Tensor
            The RnC loss.
        
        """

        z1, z2, labels = self.check_params(z1, z2, labels)
        features = torch.cat([z1, z2], dim=0)  # [2bs, feat_dim]
        labels = labels.repeat(2, 1)  # [2bs, label_dim]

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()
        return loss
    

    def check_params(self, z1, z2, labels):
        if not isinstance(labels, torch.Tensor):
            raise ValueError(f"labels must be torch.Tensor, got {type(labels)}")
        
        if not isinstance(z1, torch.Tensor) or not isinstance(z2, torch.Tensor):
            raise ValueError(f"z1 and z2 must be torch.Tensor, got {type(z1), type(z2)}")

        if labels.ndim == 0:
            labels = labels[None, None]
        if labels.ndim == 1:
            labels = labels[:, None]
        elif labels.ndim > 2:
            raise ValueError(f"Labels must have 0, 1 or 2 dimensions, got {labels.ndim}.")
        return z1, z2, labels