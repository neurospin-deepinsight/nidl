:mod:`nidl.estimators`: Available estimators
============================================

.. automodule:: nidl.estimators
   :no-members:
   :no-inherited-members:

.. No relevant user manual section yet.


Introduction
------------

An estimator is an object that fits a model based on some 
training data and is capable of inferring some properties on new data. It can
be, a classifier, a clustering algorithm, a regressor or a transformer (in the
"scikit-learn" sense). All estimators implement a ``fit`` method. Behind the hood,
it inherits from a ``pytorch_lightning.LightningModule``, and thus
benefits from all the features of the ``pytorch_lightning`` library:

- Distributed multi-GPU training
- Logging and visualization
- Clear organization of the training and evaluation code
- Automatic checkpointing and early stopping
- Callback logic


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
train_dataloader torch DataLoader [(n_samples, \*)]

val_dataloader   torch DataLoader [(n_samples, \*)]
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

Base classes for all nidl estimators.

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


Self-Supervised Learning
------------------------

Self-supervised learning embedding estimators.


Embedding estimators
....................

.. currentmodule:: nidl.estimators.ssl

.. autosummary::
   :toctree: generated/
   :template: class.rst

    SimCLR
    DCL
    YAwareContrastiveLearning
    BarlowTwins
    DINO
    IJEPA
    NeuroJEPA

.. autoclasstree:: nidl.estimators.ssl
   :strict:
   :align: center

See :doc:`losses` and :doc:`projection_heads` for the losses and projection
heads used by these estimators.


Pretrained weights
------------------

All estimators inherit from :class:`~nidl.estimators.base.NidlHubMixin`: they
can be saved, shared on the Hugging Face Hub and instantiated with published
weights, as in `braindecode <https://braindecode.org>`_:

.. code-block:: python

    from nidl.estimators.ssl import SimCLR

    model = SimCLR.from_pretrained("nidl-neurospin/brainiac-vit-b")
    transform = model.get_preprocessing()

A published estimator is a folder, or a Hub repository, holding its weights
and a ``config.json`` with everything needed to build it: the estimator, its
encoder, the shape and range of its inputs and their preprocessing. The
estimator then knows the input it expects:
:meth:`~nidl.estimators.base.NidlHubMixin.get_preprocessing` gives the
preprocessing of the training data, and batches of the wrong shape or range
raise an error in :meth:`transform`. Save your own estimators with
:meth:`~nidl.estimators.base.NidlHubMixin.save_pretrained` or
:meth:`~nidl.estimators.base.NidlHubMixin.push_to_hub`. Install the
dependencies with ``pip install nidl[pretrained]``.

The available weights are listed in the documentation of each estimator, for
now :class:`~nidl.estimators.ssl.NeuroJEPA`,
:class:`~nidl.estimators.ssl.YAwareContrastiveLearning` and
:class:`~nidl.estimators.ssl.SimCLR` (BrainIAC and Neuro-SimCLR). They keep
their own license. The BrainIAC, Neuro-SimCLR and y-Aware ones are hosted by
the `nidl-neurospin <https://huggingface.co/nidl-neurospin>`_ organization.
The Neuro-JEPA ones are gated (CC-BY-NC-ND 4.0), must be requested on their
`model page <https://huggingface.co/NYUMedML/Neuro-JEPA>`_: they cannot be
published again, so they are downloaded from the repository of their authors
and renamed when loaded, see :mod:`nidl.estimators.external`.

.. currentmodule:: nidl.estimators

.. autosummary::
   :toctree: generated/
   :template: class.rst

    base.NidlHubMixin
    external.ExternalWeights
    external.NeuroJEPAWeights


Probing estimators
------------------

Probing estimators, which are used to evaluate the quality of the representations
learned by an embedding estimator.

.. currentmodule:: nidl.estimators.probes

.. autosummary::
   :toctree: generated/
   :template: class.rst

    ModelProbing

Autoencoders
------------

Autoencoder estimators and losses.

Embedding estimators
....................

.. currentmodule:: nidl.estimators.autoencoders

.. autosummary::
   :toctree: generated/
   :template: class.rst

    VAE

.. autoclasstree:: nidl.estimators.autoencoders
   :strict:
   :align: center

See :doc:`losses` for the losses used by these estimators.


Dummy estimator
---------------

A dummy estimator that does not perform any training. This can be useful for testing purposes.

.. currentmodule:: nidl.estimators.dummy

.. autosummary::
   :toctree: generated/
   :template: class.rst

    DummyEmbeddingEstimator