.. -*- mode: rst -*-


0.0.0.dev
=========

**Released Planed Spetemeber 2025**

HIGHLIGHTS
----------

- :bdg-success:`API` Propose an API for estimators (:class:`nidl.estimators.base.BaseEstimator`), callbacks (:class:`lightning.pytorch.callbacks.Callback`), backbones (:class:`torch.nn.Module`), and transforms (:class:`nidl.transforms.Transform`).
- :bdg-primary:`Doc` Create doc with `furo <https://github.com/pradyunsg/furo>`_.

NEW
---

- :bdg-dark:`Code` Two new SSL estimatores -
  :class:`~nidl.estimators.ssl.SimCLR`,
  :class:`~nidl.estimators.ssl.YAwareContrastiveLearning`.
- :bdg-dark:`Code` Four new volume backbones -
  :class:`~nidl.volume.backbones.AlexNet`,
  :class:`~nidl.volume.backbones.DenseNet`,
  :class:`~nidl.volume.backbones.ResNet`,
  :class:`~nidl.volume.backbones.ResNetTruncated`.
- :bdg-dark:`Code` Three new generic datasets -
  :class:`~nidl.datasets.BaseImageDataset`,
  :class:`~nidl.datasets.BaseNumpyDataset`,
  :class:`~nidl.datasets.ImageDataFrameDataset`.
- :bdg-dark:`Code` The OpenBHB dataset -
  :class:`~nidl.datasets.OpenBHB`.
- :bdg-dark:`Code` A check typing callback -
  :class:`~nidl.callbacks.BatchTypingCallback`.
- :bdg-dark:`Code` Six new volume augmentations - 
  :class:`~nidl.volume.transforms.augmentation.RandomGaussianBlur`,
  :class:`~nidl.volume.transforms.augmentation.RandomGaussianNoise`,
  :class:`~nidl.volume.transforms.augmentation.RandomErasing`,
  :class:`~nidl.volume.transforms.augmentation.RandomResizedCrop`,
  :class:`~nidl.volume.transforms.augmentation.RandomRotation`
  :class:`~nidl.volume.transforms.augmentation.RandomFlip`.

Fixes
-----

Enhancements
------------

Changes
-------

