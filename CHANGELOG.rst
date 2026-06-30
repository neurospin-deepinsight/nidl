.. -*- mode: rst -*-


0.0.0.dev
=========

**Released Planed Spetemeber 2025**

HIGHLIGHTS
----------

- :bdg-success:`API` Propose an API for estimators (:class:`nidl.estimators.base.BaseEstimator`), callbacks (:class:`lightning.pytorch.callbacks.Callback`), backbones (:class:`torch.nn.Module`), and transforms (:class:`nidl.transforms.transforms.Transform`).
- :bdg-primary:`Doc` Create doc with `furo <https://github.com/pradyunsg/furo>`_.

NEW
---

- :bdg-dark:`Code` Two new SSL estimatores -
  :class:`~nidl.estimators.ssl.SimCLR`,
  :class:`~nidl.estimators.ssl.YAwareContrastiveLearning`.
- :bdg-dark:`Code` Four new volume backbones -
  :class:`~nidl.backbones.volume.AlexNet`,
  :class:`~nidl.backbones.volume.DenseNet`,
  :class:`~nidl.backbones.volume.ResNet`,
  :class:`~nidl.backbones.volume.ResNetTruncated`.
- :bdg-dark:`Code` Three new generic datasets -
  :class:`~nidl.datasets.BaseImageDataset`,
  :class:`~nidl.datasets.BaseNumpyDataset`,
  :class:`~nidl.datasets.ImageDataFrameDataset`.
- :bdg-dark:`Code` The OpenBHB dataset -
  :class:`~nidl.datasets.OpenBHB`.
- :bdg-dark:`Code` A check typing callback -
  :class:`~nidl.callbacks.BatchTypingCallback`.
- :bdg-dark:`Code` Six new volume augmentations - 
  :class:`~nidl.transforms.volume.augmentation.RandomGaussianBlur`,
  :class:`~nidl.transforms.volume.augmentation.RandomGaussianNoise`,
  :class:`~nidl.transforms.volume.augmentation.RandomErasing`,
  :class:`~nidl.transforms.volume.augmentation.RandomResizedCrop`,
  :class:`~nidl.transforms.volume.augmentation.RandomRotation`
  :class:`~nidl.transforms.volume.augmentation.RandomFlip`.

Fixes
-----

Enhancements
------------

Changes
-------

