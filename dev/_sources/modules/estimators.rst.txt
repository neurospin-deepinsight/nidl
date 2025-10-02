:mod:`nidl.estimators`: Available estimators
============================================

.. automodule:: nidl.estimators
   :no-members:
   :no-inherited-members:

.. No relevant user manual section yet.


Introduction
------------

An estimator is an object that fits a model based on some training data and
is capable of inferring some properties on new data. It can be, a classifier,
a clustering, a regressor or a transformer. All estimators implement the
fit method.


Instanciation
.............

The estimator :meth:`~BaseEstimator.__init__` method might accept
constants as arguments that
determine the estimator's behavior (like the hyperparameters and training
settings). It should not, however, take the actual training data as an
argument, as this is left to the :meth:`~BaseEstimator.fit` method.


Fitting
.......

The next thing you will probably want to do is to estimate some parameters
in the model. This is implemented in the :meth:`~BaseEstimator.fit` method,
and it's where the training happens.

The :meth:`~BaseEstimator.fit` method takes the following training data
as arguments:

================ ======================================================
Parameters
================ ======================================================
train_dataloader torch DataLoader [(n_samples, n_channels, n_features)]

val_dataloader   torch DataLoader [(n_samples, n_channels, n_features)]
================ ======================================================

Build as a ``LightningModule``, the :meth:`~BaseEstimator.fit` method gets
organized under a :meth:`~BaseEstimator.training_step` and
:meth:`~BaseEstimator.validation_step` methods.


Estimator types
...............

The proposed types of estimators are transformers, classifiers, regressors,
and clustering algorithms.

**Transformers** inherit from :class:`~base.TransformerMixin`, and implement a
:meth:`~BaseEstimator.transform` method. These are estimators which take the
input, and transform it in some way. Note that they should never change the
number of input samples, and the output of transform should correspond to its
input samples in the same given order.

**Regressors** inherit from :class:`~base.RegressorMixin`, and implement a
:meth:`~BaseEstimator.predict()` method returning the values assigned to
newly given samples. In this case the training data must returns two tensors.

**Classifiers** inherit from :class:`~base.ClassifierMixin`, and implement a
:meth:`~BaseEstimator.predict()` method returning the labels assigned to
newly given samples. In this case the training data must returns two tensors.

**Clustering** inherit from :class:`~base.ClusterMixin`, and implement a
:meth:`~BaseEstimator.predict()` method returning the labels assigned to
newly given samples. In this case the training data must returns two tensors.

Build as a ``LightningModule``, the :meth:`~BaseEstimator.transform` and the
:meth:`~BaseEstimator.predict` method gets organized under a the
:meth:`~BaseEstimator.transform_step` and
the :meth:`~BaseEstimator.predict_step` methods.


Base Classes
------------

Base classes for all estimators and various utility functions.

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


Self Supervised Learning
------------------------

Self supervised learning estimators, losses and associated tools.


Estimators
..........

.. currentmodule:: nidl.estimators.ssl

.. autosummary::
   :toctree: generated/
   :template: class.rst

    SimCLR
    YAwareContrastiveLearning

.. autoclasstree:: nidl.estimators.ssl
   :strict:
   :align: center


Losses
......

.. currentmodule:: nidl.losses

.. autosummary::
   :toctree: generated/
   :template: class.rst

    InfoNCE
    YAwareInfoNCE
    KernelMetric


Tools
.....

.. currentmodule:: nidl.estimators.ssl.utils

.. autosummary::
   :toctree: generated/
   :template: class.rst

    ProjectionHead
    SimCLRProjectionHead
    YAwareProjectionHead
