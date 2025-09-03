##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""Common callbacks."""

from .check_typing import BatchTypingCallback
from .model_probing import (
    ClassificationProbingCallback,
    ModelProbing,
    RegressionProbingCallback,
)
from .multitask_probing import MultiTaskEstimator, MultitaskModelProbing
