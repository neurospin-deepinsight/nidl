##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

""" This modules details the public API you should use and implement for a
nidl compatible dataset, as well as the datasets available in nidl.
"""

from .base import BaseImageDataset, BaseNumpyDataset
from .openbhb import OpenBHB
from .pandas_dataset import ImageDataFrameDataset
