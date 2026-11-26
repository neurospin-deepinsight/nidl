:mod:`nidl.backbones.volume`: Available backbones
=================================================

.. automodule:: nidl.backbones.volume
   :no-members:
   :no-inherited-members:

.. No relevant user manual section yet.


Introduction
------------

An backbone is a :class:`torch.nn.Module` class and contains definitions
of models for addressing different tasks, such as image classification,
image segmentation, representation learning.

Pre-trained weights
...................

nidl offers pre-trained weights for every provided architecture. The weights
can be restored using the :meth:`nidl.utils.hub.Weights.load_pretrained` method.
Available weights are listed `here <https://huggingface.co/nidl-neurospin>`_.


Weights
-------

Classes that allow to restore pre-trained weights.

.. currentmodule:: nidl.utils.hub

.. autosummary::
   :toctree: generated/
   :template: class.rst

    Weights


Volume
------

Classes that implement architectures that can be applied on brain volumes and
various utility functions.

These architectures record the parameters given to their constructor (see
:class:`~nidl.utils.config.ConfigMixin`): an estimator using one of them as
encoder can be saved with
:meth:`~nidl.utils.hub.HubMixin.save_pretrained`.

.. currentmodule:: nidl.backbones.volume

.. autosummary::
   :toctree: generated/
   :template: class.rst

    AlexNet
    DenseNet
    ResNet
    ResNetTruncated
    VisionTransformer3D
    VisionTransformer3DMoE
    MoEParams

.. autoclasstree:: nidl.backbones.volume
   :strict:
   :align: center

.. autosummary::
   :toctree: generated/
   :template: function.rst

   densenet121
   resnet18_trunc
   resnet50
   resnet50_trunc
   vit_small_patch16_128
   vit_base_patch16_128
   vit_large_patch16_128


Input adapters
..............

Classes to patchify input volumes or feature maps into sequences of tokens.

.. currentmodule:: nidl.backbones.volume.utils

.. autosummary::
   :toctree: generated/
   :template: class.rst

    SimpleFeaturesInputAdapter
    IdentityAdapter
    Patched3DInputAdapter


Surface
-------

Classes that implement architectures that ccan be applied on brain surfaces and
various utility functions.

**coming soon**
