:mod:`nidl.volume.backbones`: Available backbones
=================================================

.. automodule:: nidl.volume.backbones
   :no-members:
   :no-inherited-members:

.. No relevant user manual section yet.


Introduction
------------

An backbone is a :class:`torch.nn.Module` class and contains definitions
of models for addressing different tasks, including image classification,
image segmentation, represnetation learning...

Pre-trained weights
...................

nidl offers pre-trained weights for every provided architecture. The weights
can be restored using the :meth:`nidl.utils.Weights.load_pretrained` method.
Available weights are listed `here <https://huggingface.co/neurospin>`_.


Weights
-------

Classes that allow to restore pre-trained weights.

.. currentmodule:: nidl.utils

.. autosummary::
   :toctree: generated/
   :template: class.rst

    Weights


Volume
------

Classes that implement architectures that can be applied on brain volumes and
various utility functions.

.. currentmodule:: nidl.volume.backbones

.. autosummary::
   :toctree: generated/
   :template: class.rst

    AlexNet
    DenseNet
    ResNet
    ResNetTruncated

.. autoclasstree:: nidl.volume.backbones
   :strict:
   :align: center

.. autosummary::
   :toctree: generated/
   :template: function.rst

   densenet121
   resnet18_trunc
   resnet50
   resnet50_trunc


Surface
-------

Classes that implement architectures that ccan be applied on brain surfaces and
various utility functions.

**coming soon**
