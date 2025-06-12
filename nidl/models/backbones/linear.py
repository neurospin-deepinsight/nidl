import torch.nn as nn

class Linear(nn.Module):
    """
    An affine linear layer applied to the input data.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        bias : bool, optional
            Whether to include a bias term in the linear layer. Default is True.
        """
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)