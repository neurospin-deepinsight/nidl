
from torch import nn
import math

class AlexNet(nn.Module):
    """
    Model implemented in https://doi.org/10.5281/zenodo.4309677 by Abrol et al., 2021
    Small adaptation for representation learning: the final regressor head is replaced by simple linear layer.
    """
    def __init__(self, n_embedding=128, n_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(n_channels, 64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool3d(1),
        )
        self.regressor = nn.Linear(128, n_embedding)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        xp = self.features(x)
        x = xp.view(xp.size(0), -1)
        x = self.regressor(x)
        return x