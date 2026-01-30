:mod:`nidl.metrics`: Available metrics
======================================

.. automodule:: nidl.metrics
   :no-members:
   :no-inherited-members:

.. No relevant user manual section yet.


Introduction
------------

A metric is an object (most likely a function) that allows you to compute
standard scores, usually not natively present in :mod:`sklearn.metrics`
or in `torchmetrics`. Most metrics in nidl handle both :class:`numpy.ndarray`
and :class:`torch.Tensor` and returns an output consistent with the input.  


Regression metrics
------------------

Functions for all regression metrics.

.. currentmodule:: nidl.metrics

.. autosummary::
   :toctree: generated/
   :template: function.rst

    pearson_r

Self-supervised metrics
-----------------------

Functions for all self-supervised metrics.

.. currentmodule:: nidl.metrics

.. autosummary::
   :toctree: generated/
   :template: function.rst

    alignment_score
    uniformity_score
    contrastive_accuracy_score

