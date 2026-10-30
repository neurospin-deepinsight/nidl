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
We propose in nidl original monitoring callbacks as well as
neuroimaging focused callbacks.


.. autoclasstree:: nidl.callbacks
   :strict:
   :align: center


Monitoring
----------

Classes for all monitoring callbacks.

.. currentmodule:: nidl.callbacks

.. autosummary::
   :toctree: generated/
   :template: class.rst

    BatchTypingCallback


Neuroimaging
------------

Classes for all neuroimaging callbacks.

.. currentmodule:: nidl.callbacks

.. autosummary::
   :toctree: generated/
   :template: class.rst

    BatchTypingCallback
    ClassificationProbingCallback
    RegressionProbingCallback
    MultitaskModelProbing
    ModelProbing
