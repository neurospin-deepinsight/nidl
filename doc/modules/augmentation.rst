:mod:`nidl.transforms`: Available augmentations
===============================================

.. automodule:: nidl.transforms
   :no-members:
   :no-inherited-members:

.. No relevant user manual section yet.


Introduction
------------

A transform is an object that can be called on some data and is capable of
modifying some properties to generate new data.


Instanciation
.............

The transform :meth:`~Transform.__init__` method only accepts the
probability that this transform will be applied. It should not take the data
as an argument, as this is left to the :meth:`~Transform.__call__` method.


Composability
.............

Transforms can be composed using the :class:`torchvision.transforms.Compose`
class to create directed acyclic graphs defining the probability that each
transform will be applied.


Reproducibility
...............

When transforms are instantiated, we typically need to pass values that
will be used to sample the transform parameters when the
:meth:`~Transform.__call__` method of the transform is called, i.e., when
the transform instance is called.

All random transforms have a ``seed`` parameter to have a corresponding
deterministic behaviour.


Base Classes
------------

Base classes for all augmentations and various utility functions.

.. currentmodule:: nidl.transforms

.. autosummary::
   :toctree: generated/
   :template: class.rst

    Transform
    Identity
    MultiViewsTransform
    VolumeTransform


Volume
------

Classes that implement augmentations on brain 3D volumes and various utility
functions.

.. currentmodule:: nidl.volume.transforms.augmentation

.. autosummary::
   :toctree: generated/
   :template: class.rst

    RandomGaussianBlur
    RandomGaussianNoise
    RandomErasing
    RandomFlip
    RandomResizedCrop
    RandomRotation

.. autoclasstree:: nidl.volume.transforms.augmentation
   :strict:
   :align: center


Surface
-------

Classes that implement augmentations on brain surface and various utility
functions.

**coming soon**
