##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

""" Common losses.
"""

from .infonce import InfoNCE
from .mcvae import MCVAELoss
from .vae import BetaHLoss
from .yaware_infonce import KernelMetric, YAwareInfoNCE
