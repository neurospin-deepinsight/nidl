Nidl
====

.. container:: index-paragraph

    Nidl is a Python library to perform distributed training and evaluation
    of deep learning models on large-scale neuroimaging data (anatomical
    volumes and surfaces, fMRI). 

    It follows the PyTorch design for the training logic and the scikit-learn
    API for the models (in particular fit, predict and transform). 

    Supervised, self-supervised and unsupervised models are available (with
    pre-trained weights) along with open datasets.

.. grid::

    .. grid-item-card:: :fas:`rocket` Quickstart
        :link: quickstart
        :link-type: ref
        :columns: 12 12 4 4
        :class-card: sd-shadow-md
        :class-title: sd-text-primary
        :margin: 2 2 0 0

        Get started with Nidl

    .. grid-item-card:: :fas:`th` Examples
        :link: auto_examples/index.html
        :link-type: url
        :columns: 12 12 4 4
        :class-card: sd-shadow-md
        :class-title: sd-text-primary
        :margin: 2 2 0 0

        Discover functionalities by reading examples

    .. grid-item-card:: :fas:`book` User guide
        :link: user_guide
        :link-type: ref
        :columns: 12 12 4 4
        :class-card: sd-shadow-md
        :class-title: sd-text-primary
        :margin: 2 2 0 0

        Learn about neuroimaging analysis

Key features
------------

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: :fas:`database` Datasets
        :class-card: sd-shadow-sm
        :class-title: sd-text-primary

        Large-scale open neuroimaging datasets ready for PyTorch, fetched on
        demand from Hugging Face and streamed during training:
        :class:`~nidl.datasets.OpenBHB` (~4k quality-checked brain T1w MRI 
        uniformly preprocessed) and :class:`~nidl.datasets.OpenMind` (114k brain MRIs). 
        Bring your own data with :class:`~nidl.datasets.ImageDataFrameDataset`.

        +++
        :ref:`Learn more <open_datasets>` :fas:`arrow-right`

    .. grid-item-card:: :fas:`brain` Models
        :class-card: sd-shadow-sm
        :class-title: sd-text-primary

        Self-supervised (:class:`~nidl.estimators.ssl.SimCLR`,
        :class:`~nidl.estimators.ssl.BarlowTwins`,
        :class:`~nidl.estimators.ssl.DCL`, :class:`~nidl.estimators.ssl.DINO`,
        :class:`~nidl.estimators.ssl.NeuroJEPA`,), weakly supervised
        (:class:`~nidl.estimators.ssl.YAwareContrastiveLearning`) and
        generative (:class:`~nidl.estimators.autoencoders.VAE`) models, all
        exposing the scikit-learn ``fit`` / ``transform`` API.

        +++
        :ref:`Learn more <self_supervised_learning>` :fas:`arrow-right`

    .. grid-item-card:: :fas:`dice` Augmentations
        :class-card: sd-shadow-sm
        :class-title: sd-text-primary

        3D-native preprocessing and data augmentation:
        :class:`~nidl.transforms.volume.augmentation.RandomResizedCrop`,
        :class:`~nidl.transforms.volume.augmentation.RandomFlip`,
        :class:`~nidl.transforms.volume.augmentation.RandomRotation`,
        :class:`~nidl.transforms.volume.augmentation.RandomGaussianNoise`, ...
        Generate multiple views per sample with
        :class:`~nidl.transforms.transforms.MultiViewsTransform`.

        +++
        :ref:`Learn more <data_augmentation>` :fas:`arrow-right`

    .. grid-item-card:: :fas:`chart-line` Evaluation
        :class-card: sd-shadow-sm
        :class-title: sd-text-primary

        Assess learned representations with any scikit-learn probe using
        :class:`~nidl.estimators.probes.ModelProbing`, or monitor them while
        training with :class:`~nidl.callbacks.ModelProbingCallback` and
        :class:`~nidl.callbacks.MetricsCallback`.

        +++
        :ref:`Learn more <model_probing>` :fas:`arrow-right`

Fit a model in a few lines
--------------------------

Models are configured like scikit-learn estimators and trained like
`Lightning <https://lightning.ai>`_ modules: any ``Trainer`` argument
(``accelerator``, ``devices``, ``strategy``, ``max_epochs``, ...) can be
given to the estimator, so going from one GPU to a multi-node cluster
requires no code change.

.. tab-set::

    .. tab-item:: Self-supervised

        .. code-block:: python

            from torch.utils.data import DataLoader
            from torchvision.transforms import Compose

            from nidl.backbones.volume import resnet18
            from nidl.datasets import OpenBHB
            from nidl.estimators.ssl import SimCLR
            from nidl.transforms import MultiViewsTransform
            from nidl.transforms.volume.augmentation import (
                RandomFlip,
                RandomResizedCrop,
            )

            # Two augmented views of each brain volume
            augment = Compose([
                RandomResizedCrop(target_shape=128, scale=(0.5, 1.0)),
                RandomFlip(axes=0, flip_probability=0.5),
            ])
            dataset = OpenBHB(
                "/data/openbhb",
                modality="vbm",
                target=None,
                transforms=MultiViewsTransform(augment, n_views=2),
            )
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            # Contrastive pre-training of a 3D ResNet-18 on 4 GPUs
            model = SimCLR(
                encoder=resnet18(n_embedding=512),
                proj_input_dim=512,
                max_epochs=100,
                accelerator="gpu",
                devices=4,
            )
            model.fit(loader)
            embeddings = model.transform(test_loader)

        See :ref:`sphx_glr_auto_examples_plot_dcl_vs_simclr.py` for a full
        example.

    .. tab-item:: Weakly supervised

        .. code-block:: python

            from nidl.estimators.ssl import YAwareContrastiveLearning

            # The dataset now returns (views, age): age guides the
            # contrastive objective through a Gaussian kernel
            dataset = OpenBHB(
                "/data/openbhb",
                modality="vbm",
                target="age",
                transforms=MultiViewsTransform(augment, n_views=2),
            )
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            model = YAwareContrastiveLearning(
                encoder=resnet18(n_embedding=512),
                proj_input_dim=512,
                bandwidth=4**2,
                max_epochs=100,
            )
            model.fit(loader)

        See :ref:`sphx_glr_auto_examples_plot_yaware_openbhb.py` for a full
        example.

    .. tab-item:: Probing

        .. code-block:: python

            from sklearn.linear_model import Ridge

            from nidl.estimators.probes import ModelProbing

            # Fit a linear probe on top of the frozen embeddings
            probing = ModelProbing(
                embedding_estimator=model,
                probe=Ridge(),
                scoring=["neg_mean_absolute_error", "r2"],
            )
            probing.fit(train_loader)
            probing.score(test_loader)
            # {'neg_mean_absolute_error': -5.1, 'r2': 0.78}

        See :ref:`sphx_glr_auto_examples_plot_model_probing.py` for a full
        example.

Backbones
---------

3D architectures are available in :mod:`nidl.backbones.volume`: ResNet,
DenseNet, AlexNet and Vision Transformers (including Mixture-of-Experts),
with :ref:`pre-trained weights <pretrained_models>` on large neuroimaging
cohorts. See the :ref:`architectures <architectures>` guide.

.. toctree::
   :hidden:
   :includehidden:
   :titlesonly:

   quickstart.md
   auto_examples/index.rst
   user_guide.rst
   modules/index.rst
   glossary.rst

.. toctree::
   :hidden:
   :caption: Development

   development.rst
   ci.rst
   maintenance.rst
   whats_new.rst
   authors.rst
   Versions <versions.rst>
   GitHub Repository <https://github.com/neurospin-deepinsight/nidl>
