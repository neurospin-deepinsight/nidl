:mod:`nidl.estimators`: Available estimators
============================================

.. automodule:: nidl.estimators
   :no-members:
   :no-inherited-members:

.. No relevant user manual section yet.

Base Classes
------------

.. currentmodule:: nidl.estimators

.. autosummary::
   :toctree: generated/
   :template: class.rst

    BaseEstimator
    ClassifierMixin
    ClusterMixin
    RegressorMixin
    TransformerMixin

.. autoclasstree:: nidl.estimators
   :strict:
   :align: center


SSL
---

.. currentmodule:: nidl.estimators.ssl

.. autosummary::
   :toctree: generated/
   :template: class.rst

    SimCLR
    YAwareContrastiveLearning

.. autoclasstree:: nidl.estimators.ssl
   :strict:
   :align: center


.. currentmodule:: nidl.estimators.ssl.utils

.. autosummary::
   :toctree: generated/
   :template: class.rst

    SimCLRProjectionHead
    YAwareProjectionHead


.. currentmodule:: nidl.losses

.. autosummary::
   :toctree: generated/
   :template: class.rst

    InfoNCE
    YAwareInfoNCE
    KernelMetric
