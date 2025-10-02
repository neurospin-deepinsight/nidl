:mod:`nidl.datasets`: Available datasets
========================================

.. automodule:: nidl.datasets
   :no-members:
   :no-inherited-members:

.. No relevant user manual section yet.


Introduction
------------

nidl offers tools to easily download publicly available neuroimaging datasets.
If you use any of them, please visit the corresponding website
(linked in each description) and make sure you comply with any data usage
agreement and you acknowledge the corresponding authors' publications.

nidl also offers tools to fetch datasets available on you local machine in a
coherent and reproducible way.


Local Fetchers
--------------

Base classes for to create a local fetcher.

.. currentmodule:: nidl.datasets

.. autosummary::
   :toctree: generated/
   :template: class.rst

   BaseImageDataset
   BaseNumpyDataset
   ImageDataFrameDataset


General Population
------------------

Classes to fetch publicly available neuroimaging datasets.

.. currentmodule:: nidl.datasets

.. autosummary::
   :toctree: generated/
   :template: class.rst

   OpenBHB
