.. _pretrained_models:

=================
Pretrained Models
=================

nidl gives access to models pretrained on large neuroimaging cohorts, ready
to be used as frozen feature extractors or fine-tuned on your own data. Each
model is published on the Hugging Face Hub with a ``config.json`` that fully
describes the nidl estimator: its parameters, its encoder, the shape of its
inputs and their preprocessing. Loading a model thus gives the exact
architecture of its original implementation, the preprocessing its weights
expect, and checks on the inputs given to it.

Two families of models are available:

- **open models**, pretrained with nidl on open data. Their data, code,
  training recipe and weights are all public, so that anyone can reproduce
  them, compare new methods with them under the same conditions, or build
  upon them;
- **published models**, ported to nidl and validated against their original
  code. They are faithful to their papers, with their own pretraining data
  and preprocessing.


Quick start
-----------

Install the dependencies with ``pip install nidl[pretrained]``, then:

.. code-block:: python

    from torch.utils.data import DataLoader
    from nidl.estimators.ssl import SimCLR

    model = SimCLR.from_pretrained("nidl-neurospin/open-simclr-resnet50")
    preprocess = model.get_preprocessing()

    loader = DataLoader([preprocess(volume) for volume in volumes], batch_size=4)
    features = model.transform(loader)  # frozen features
    # or fine-tune it: model.fit(train_loader)

:meth:`~nidl.estimators.base.NidlHubMixin.get_preprocessing` returns the
transform reproducing the preprocessing of the original code. For some
published models, its parameters fix what depends on your data, such as the
orientation of the volumes: see each model below for the expected input.
:meth:`~nidl.estimators.BaseEstimator.transform` rejects inputs of the wrong
shape. See :ref:`sphx_glr_auto_examples_plot_pretrained_openbhb.py`
for a complete pipeline on OpenBHB.


Open models
-----------

The open models are pretrained with nidl on the train split of
:class:`~nidl.datasets.OpenMind`, 787 OpenNeuro datasets with a quasi-raw
preprocessing (skull-stripping, bias field correction and linear
registration to MNI152 at 1 mm, as OpenBHB), on a single GPU.
Everything needed to rebuild them is public:

- the data, downloaded by nidl at a fixed revision;
- the estimator, its hyper-parameters and its training parameters, in the
  ``config.json`` of each model: ``from_config`` builds the same estimator,
  randomly initialized, ready to be trained;
- the data loading and the augmentations, in the
  :ref:`pretraining recipe <sphx_glr_auto_examples_plot_open_models_pretraining.py>`;
- the weights, under the CC-BY 4.0 license, with all the parameters of the
  estimator (e.g., the projection heads and the predictor of NeuroJEPA), so
  that their pretraining can also be continued.

.. pretrained-models::
   :family: open
   :view: table

.. pretrained-models::
   :family: open
   :view: cards


Published models
----------------

.. pretrained-models::
   :family: published
   :view: table

All the repositories belong to the `nidl-neurospin
<https://huggingface.co/nidl-neurospin>`_ organization. :fas:`lock` The
weights are gated: request access on the model page of their authors first.
Click on a model for its pretraining data, expected input, license and
citation.

.. pretrained-models::
   :family: published
   :view: cards


Brain age on OpenBHB
--------------------

To illustrate the use of these models, the age of the participants of
:class:`~nidl.datasets.OpenBHB` is predicted from their frozen features. 
Here, models were pretrained on different data and with different
preprocessings so this comparison is biased but offers a first idea of their
performance. The open models share the same pretraining data and
preprocessing, so that they can be compared with each other.

.. pretrained-models::
   :view: openbhb

.. image:: images/plot_pretrained_openbhb_age.png
   :alt: True and predicted ages of the OpenBHB test participants, for each
         pretrained model.


Share your own model
--------------------

Any nidl estimator can be saved in the same format with
:meth:`~nidl.estimators.base.NidlHubMixin.save_pretrained`, or published
directly with :meth:`~nidl.estimators.base.NidlHubMixin.push_to_hub`:

.. code-block:: python

    model.fit(train_loader)
    model.push_to_hub(
        "my-org/my-model",
        preprocessing={"transform": "ZNormalization", "kwargs": {}},
        input_shape=(1, 121, 145, 121),
    )

It can then be loaded by anyone with ``from_pretrained("my-org/my-model")``.
The training parameters of the estimator, such as ``max_epochs`` or
``precision``, are saved with it, so that ``from_config`` rebuilds an
estimator trained the same way: share your data loading and augmentations as
the open models do to make your model fully reproducible.
For weights published elsewhere that cannot be redistributed, implement an
:class:`~nidl.estimators.external.ExternalWeights` converter.


Licenses and access
-------------------

The open models are released under the CC-BY 4.0 license. The other
weights keep the license of their authors, given above and in the model
card of each repository: check it before using a model, in particular for
commercial or clinical use. Weights that cannot be redistributed, such as the
Neuro-JEPA ones, are downloaded from the repository of their authors, at the
commit validated with nidl, and only renamed when loading. The nidl
repositories then only hold their configuration, see
:class:`~nidl.estimators.external.ExternalWeights`.