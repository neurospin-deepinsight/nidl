:mod:`nidl.callbacks`: Available callbacks
==========================================

.. automodule:: nidl.callbacks
   :no-members:
   :no-inherited-members:

.. No relevant user manual section yet.


Introduction
------------

A callback is a :class:`lightning.pytorch.callbacks.Callback` class that allows
you to add arbitrary self-contained programs to your training. At specific
points during the flow of execution (hooks), the callback interface allows you
to design programs that encapsulate a full set of functionality. It de-couples
functionality that does not need to be in the lightning module and can be
shared across projects.

Lightning provides a large set of callbacks described
`here <https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html>`_.
We propose in nidl critical callbacks for model probing and metrics
computation. It allows a better decoupling of the model training and evaluation
logic from the actual implementation of these functionalities.


.. autoclasstree:: nidl.callbacks
   :strict:
   :align: center


Probing callbacks
-----------------

Classes for all callbacks performing model's probing.

.. currentmodule:: nidl.callbacks

.. autosummary::
   :toctree: generated/
   :template: class.rst

    ClassificationProbingCallback
    RegressionProbingCallback
    MultitaskModelProbing
    ModelProbing


Metrics callback
----------------

Classes for all callbacks performing metrics computation.

.. currentmodule:: nidl.callbacks

.. autosummary::
   :toctree: generated/
   :template: class.rst

    MetricsCallback

Typing callback
----------------

Classes for all callbacks checking the batch format given by a dataloader
against the expected type.

.. currentmodule:: nidl.callbacks

.. autosummary::
   :toctree: generated/
   :template: class.rst

    BatchTypingCallback
