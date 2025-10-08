##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

""" This modules details the public API you should use and implement for a
nidl compatible backbone, as well as the backbones available in nidl.
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
