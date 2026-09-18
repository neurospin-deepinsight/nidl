:mod:`nidl.estimators.ssl.utils`: Available projection heads
===============================================================

.. automodule:: nidl.estimators.ssl.utils
   :no-members:
   :no-inherited-members:

.. No relevant user manual section yet.


Introduction
------------

A projection head is a small :class:`torch.nn.Module` mapping the
representations produced by an encoder into the space where a
self-supervised loss (see :doc:`losses`) is applied. It is typically
discarded after pre-training, once the encoder is used for downstream tasks.


Projection heads
-----------------

.. currentmodule:: nidl.estimators.ssl.utils

.. autosummary::
   :toctree: generated/
   :template: class.rst

    ProjectionHead
    SimCLRProjectionHead
    YAwareProjectionHead
    BarlowTwinsProjectionHead
    DINOProjectionHead

.. autoclasstree:: nidl.estimators.ssl.utils
   :strict:
   :align: center
