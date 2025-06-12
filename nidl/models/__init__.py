# Supervised models
from .supervised.rnc import RnC
from .supervised.regression import DeepRegressor

# Self-supervised models
from .ssl.simclr import SimCLR
from .ssl.vicreg import VICReg
from .ssl.yaware import yAware

__all__ = [
    "RnC",
    "DeepRegressor",
    "SimCLR",
    "VICReg",
    "yAware",
]