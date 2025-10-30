##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


# Module current version
version_major = 0
version_minor = 0
version_micro = 1

# Expected by setup.py: string of form "X.Y.Z"
__version__ = f"{version_major}.{version_minor}.{version_micro}"

# Project descriptions
NAME = "nidl"
DESCRIPTION = """
Deep learning for NeuroImaging in Python.
"""
SUMMARY = """
.. container:: summary-carousel

    `nidl` is a Python module that includes the follwoing unified plugins:

    * **surfify**: deep leaning in the cortical surface.
    * **...**: ...
"""
LONG_DESCRIPTION = (
    "Nidl provides unified deep learning interfaces (compatible with `PyTorch "
    "Lightning <https://lightning.ai/pytorch-lightning>`_) to analyze brain "
    "volumes and surfaces. It also provides a way to describe your "
    "experiments using a single configuration file (for production, a "
    "more advanced tool is availble in `Hydra <https://hydra.cc>`_).\n")
LINKS = {"surfify": "https://github.com/neurospin-deepinsight/surfify"}
URL = "https://github.com/neurospin-deepinsight/nidl"
AUTHOR = """
nidl developers
"""
