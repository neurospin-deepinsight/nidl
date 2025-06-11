import torch.nn as nn

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
        A series of FC-BN-ReLU parametrized by a list of int "layers"
    """
    def __init__(self, layers, n_embedding):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(*[BasicBlock(layers[i], layers[i+1]) for i in range(len(layers)-1)],
                                 nn.Linear(layers[-1], n_embedding))

    def forward(self, x):
        x = x.squeeze(dim=1)
        x = self.mlp(x)
        return x