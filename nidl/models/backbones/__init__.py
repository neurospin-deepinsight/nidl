""" Backbones generic to all NIDL models."""

from .mlp import MLP
from .linear import Linear

__all__ = [
    "MLP",
    "Linear",
]