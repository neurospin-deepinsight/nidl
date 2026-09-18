:mod:`nidl.losses`: Available losses
=====================================

.. automodule:: nidl.losses
   :no-members:
   :no-inherited-members:

.. No relevant user manual section yet.


Introduction
------------

A loss is a :class:`torch.nn.Module` (or a plain callable) implementing the
objective function optimized during the ``training_step`` of an estimator.
Losses are decoupled from the estimators that use them so that they can be
reused, benchmarked or swapped independently.


Self-supervised learning losses
--------------------------------

Losses used by the self-supervised learning embedding estimators (see
:ref:`modules`).

.. currentmodule:: nidl.losses

.. autosummary::
   :toctree: generated/
   :template: class.rst

    InfoNCE
    DCLLoss
    DCLWLoss
    YAwareInfoNCE
    BarlowTwinsLoss
    DINOLoss


Autoencoder losses
-------------------

Losses used by the autoencoder estimators.

.. currentmodule:: nidl.losses

.. autosummary::
   :toctree: generated/
   :template: class.rst

    BetaVAELoss
