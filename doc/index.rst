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

Featured works
--------------

.. grid::

  .. grid-item-card::
    :link: auto_examples/plot_openbhb.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    .. grid::
      :gutter: 3
      :margin: 0
      :padding: 0

      .. grid-item::
        :columns: 12 4 4 4

        .. image:: _images/sphx_glr_simclr_stl10_thumb.png

      .. grid-item::
        :columns: 12 8 8 8

        .. div:: sd-font-weight-bold

          OpenBHB dataset

        B Dufumier et al.: `OpenBHB - a Large-Scale Multi-Site Brain MRI
        Dataset for Age Prediction and Debiasing
        <https://doi.org/10.1016/j.neuroimage.2022.119637>`_, NeuroImage
        2022.

  .. grid-item-card::
    :link: auto_examples/simclr_stl10.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    .. grid::
      :gutter: 3
      :margin: 0
      :padding: 0

      .. grid-item::
        :columns: 12 4 4 4

        .. image:: _images/sphx_glr_simclr_stl10_thumb.png

      .. grid-item::
        :columns: 12 8 8 8

        .. div:: sd-font-weight-bold

          SimCLR

        T Chen et al.: `A Simple Framework for Contrastive Learning of Visual 
        Representations
        <https://proceedings.mlr.press/v119/chen20j/chen20j.pdf>`_, 
        ICML 2020.

  .. grid-item-card::
    :link: auto_examples/plot_barlowtwins_openbhb.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    .. grid::
      :gutter: 3
      :margin: 0
      :padding: 0

      .. grid-item::
        :columns: 12 4 4 4

        .. image:: _images/sphx_glr_simclr_stl10_thumb.png

      .. grid-item::
        :columns: 12 8 8 8

        .. div:: sd-font-weight-bold

          Barlow Twins

        Zbonta et al.: `Barlow Twins, Self-Supervised Learning via Redundancy Reduction
        <http://proceedings.mlr.press/v139/zbontar21a/zbontar21a.pdf>`_, 
        PMLR 2021.
        
  .. grid-item-card::
    :link: auto_examples/plot_yaware_openbhb.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    .. grid::
      :gutter: 3
      :margin: 0
      :padding: 0

      .. grid-item::
        :columns: 12 4 4 4

        .. image:: _images/sphx_glr_simclr_stl10_thumb.png

      .. grid-item::
        :columns: 12 8 8 8

        .. div:: sd-font-weight-bold

          y-Aware weakly supervised learning

        B Dufumier et al.: `Exploring the potential of representation and
        transfer learning for anatomical neuroimaging - Application to
        psychiatry <https://doi.org/10.1016/j.neuroimage.2024.120665>`_,
        NeuroImage 2024.

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
