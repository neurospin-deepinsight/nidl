import torch.nn as nn
from typing import List


class BasicBlock(nn.Module):
    def __init__(self, inp, out):
        super(BasicBlock, self).__init__()
        self.linear = nn.Linear(inp, out)
        self.bn = nn.BatchNorm1d(out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MLP(nn.Module):
    """
        A series of FC-BN-ReLU layers followed by a final linear layer.
    """
    def __init__(self, layers: List[int], n_embedding: int):
        """
        Parameters
        ----------
        layers : list of int
            List of integers representing the number of neurons in each layer.
        n_embedding : int
            The number of output features for the final layer.
        """
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(*[BasicBlock(layers[i], layers[i+1]) for i in range(len(layers)-1)],
                                 nn.Linear(layers[-1], n_embedding))

    def forward(self, x):
        x = x.squeeze(dim=1)
        x = self.mlp(x)
        return x