##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

""" Definition of common volumic architectures.
"""

from .alexnet3d import AlexNet
from .densenet3d import (
    DenseNet,
    densenet121,
)
from .resnet3d import (
    ResNet,
    ResNetTruncated,
    resnet18,
    resnet18_trunc,
    resnet50,
    resnet50_trunc,
)
from .vae import (
    VAE,
)
