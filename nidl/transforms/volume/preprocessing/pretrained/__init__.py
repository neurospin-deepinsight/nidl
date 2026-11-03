##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""Preprocessing expected by published pretrained weights.

Each transform reproduces the preprocessing of the original implementation
of a model, so that its weights produce meaningful features. They are built
from the generic nidl transforms whenever these give the exact same output.
"""

from .brainiac import BrainIACPreprocessing
from .neuro_simclr import NeuroSimCLRPreprocessing
from .neurojepa import NeuroJEPAPreprocessing
