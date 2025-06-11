import torch.nn as nn

"""
    Similarity metric between embeddings for self-supervised learning. It includes simple 
    cosine similarity.
"""

class PairwiseCosineSimilarity(nn.Module):

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, X1, X2):
        X1 = nn.functional.normalize(X1, p=2, dim=self.dim)
        X2 = nn.functional.normalize(X2, p=2, dim=self.dim)
        return X1 @ X2.T

