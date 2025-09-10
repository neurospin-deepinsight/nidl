.. -*- mode: rst -*-

.. image:: https://img.shields.io/badge/python-3.9%20%7C%203.12-blue
    :target: https://img.shields.io/badge/python-3.9%20%7C%203.12-blue
    :alt: Python Version

.. image:: https://coveralls.io/repos/github/neurospin-deepinsight/nidl/badge.svg?branch=main
    :target: https://coveralls.io/github/neurospin-deepinsight/nidl
    :alt: Coverage Status

.. image:: https://github.com/neurospin-deepinsight/nidl/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/neurospin-deepinsight/nidl/actions
    :alt: Github Actions Test Status

.. image:: https://github.com/neurospin-deepinsight/nidl/actions/workflows/linters.yml/badge.svg
    :target: https://github.com/neurospin-deepinsight/nidl/actions
    :alt: Github Actions Linter Status

.. image:: https://github.com/neurospin-deepinsight/nidl/actions/workflows/documentation.yml/badge.svg
    :target: http://neurospin-deepinsight.github.io/nidl
    :alt: Github Actions Doc Build Status

.. image:: https://badge.fury.io/py/nidl.svg
    :target: https://pypi.org/project/nidl
    :alt: Pypi Package

nidl
====

Nidl is a Python library to perform distributed training and evaluation
of deep learning models on large-scale neuroimaging data (anatomical
volumes and surfaces, fMRI). 

It follows the PyTorch design for the training logic and the scikit-learn
API for the models (in particular fit, predict and transform). 

Supervised, self-supervised and unsupervised models are available (with
pre-trained weights) along with open datasets.


Important links
===============

- Official source code repo: https://github.com/neurospin-deepinsight/nidl
- HTML documentation (stable release): https://neurospin-deepinsight.github.io/nidl


Install
=======

Latest release
--------------

**1. Setup a virtual environment**

We recommend that you install ``nidl`` in a virtual Python environment,
either managed with the standard library ``venv`` or with ``conda``.
Either way, create and activate a new python environment.

With ``venv``:

.. code-block:: bash

    python3 -m venv /<path_to_new_env>
    source /<path_to_new_env>/bin/activate

Windows users should change the last line to ``\<path_to_new_env>\Scripts\activate.bat``
in order to activate their virtual environment.

With ``conda``:

.. code-block:: bash

    conda create -n nidl python=3.12
    conda activate nidl

**2. Install nidl with pip**

Execute the following command in the command prompt / terminal
in the proper python environment:

.. code-block:: bash

    python3 -m pip install -U nidl


Check installation
------------------

Try importing nidl in a python / iPython session:

.. code-block:: python

    import nidl

If no error is raised, you have installed nidl correctly.


Where to start
==============

Examples are available in the `gallery <https://neurospin-deepinsight.github.io/nidl/auto_gallery/index.html>`_.


Dependencies
============

The required dependencies to use the software are listed
in the file `pyproject.toml <https://github.com/neurospin-deepinsight/nidl/blob/main/pyproject.toml>`_.
