Introduction
============


What is ``nidl``?
=================

``nidl`` is a Python library to perform distributed training and evaluation
of deep learning models on large-scale neuroimaging data (anatomical
volumes and surfaces, fMRI). 

It follows the PyTorch design for the training logic and the scikit-learn
 API for the models (in particular fit, predict and transform). 

:ref:`Supervised <supervised_learning>`, :ref:`self-supervised <self_supervised_learning>` and
unsupervised models are available (with
pre-trained weights) along with open datasets. 


.. note::

    It is ok if these terms don't make sense to you yet:
    this guide will walk you through them in a comprehensive manner.


.. _quick_start:


Using ``nidl`` for the first time
==================================

``nidl`` is a Python library. If you have never used Python before,
you should probably have a look at a `general introduction about Python <https://www.learnpython.org/>`_
as well as to `Scientific Python Lectures <https://lectures.scientific-python.org/>`_ before diving into ``nidl``.

First steps with ``nidl``
-------------------------

At this stage, you should have :ref:`installed <quickstart>` ``nidl`` and
opened a Jupyter notebook or an IPython / Python session.  First, load
``nidl`` with

.. code-block:: python

    import nidl

``nidl`` comes in with some data that are commonly used in neuroimaging.


Learning with the API references
--------------------------------

In the last commands, you just made use of ``nidl`` modules.
All modules are described in the :ref:`API references <modules>`.

Oftentimes, if you are already familiar with the problems and vocabulary of
MRI analysis, the module and function names are explicit enough that you
should understand what ``nidl`` does.

Learning with examples
----------------------

``nidl`` comes with a lot of :ref:`examples <examples>`.
Going through them should give you a precise overview of what you can achieve
with this package.

For new-comers, we recommend going through the following examples in the
suggested order:

.. raw:: html

    <div class="sphx-glr-thumbnails">

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A simple example...">

.. only:: html

    .. image:: _images/sphx_glr_simclr_stl10_thumb.png
        :alt: Basic nidl example: Playing with estimators

    :ref:`sphx_glr_auto_examples_simclr_stl10.py`

.. raw:: html

    <div class="sphx-glr-thumbnail-title">
        Basic nidl example: Playing with estimators
    </div>
    </div>

.. raw:: html

    </div>


Finding help
------------

On top of this guide, there is a lot of content available outside of ``nidl``
that could be of interest to new-comers:

1.  `Handbook of Functional MRI Data Analysis <https://www.cambridge.org/be/universitypress/subjects/statistics-probability/statistics-life-sciences-medicine-and-health/handbook-functional-mri-data-analysis>`_
    by Russel Poldrack, Jeanette Mumford and Thomas Nichols.

2.  The documentation of ``scikit-learn`` explains each method with tips on practical use and examples: :sklearn:`\ `.
    While not specific to neuroimaging, it is often a recommended read.

3.  (For Python beginners) A quick and gentle introduction to scientific computing
    with Python with the `scientififc Python lectures <https://lectures.scientific-python.org/>`_.
    Moreover, you can use ``nidl`` with `Jupyter <https://jupyter.org/>`_ notebooks or
    `IPython <https://ipython.org/>`_ sessions. They provide an interactive
    environment that greatly facilitates debugging and visualization.

Besides, you can find help on :neurostars:`neurostars <>` for questions
related to ``nidl`` and to computational neuroscience in general.
We can also be reached on :nidl-gh:`github <issues>` in case you find a bug.

Applications to Neuroimaging
============================

``nidl`` brings easy-to-use deep learning tools that can be leveraged to
solve more complex applications.
The interested reader can dive into the following articles for more content.

We give a non-exhaustive list of such important applications.

**Diagnosis and prognosis**

Predicting a clinical score or even treatment response
from brain imaging with :ref:`supervised
learning <supervised_learning>` e.g. :footcite:t:`Mourao-miranda2012`.

**Information mapping**

Using the prediction accuracy of a classifier
to characterize relationships between brain images and stimuli. (e.g.
:footcite:t:`Kriegeskorte2006`)

**Transfer learning**

Measuring how much an estimator trained on one
specific psychological process/task can predict the neural activity
underlying another specific psychological process/task
(e.g. discriminating left from
right eye movements also discriminates additions from subtractions :footcite:p:`Knops2009`)

**High-dimensional multivariate statistics**

From a statistical point of view, machine learning implements
statistical estimation of models with a large number of parameters.
Tricks pulled in machine learning (e.g. regularization) can
make this estimation possible despite the usually
small number of observations in the neuroimaging domain :footcite:p:`Varoquaux2012`.
This usage of machine learning requires some understanding of the models.

**Data mining / exploration**

Data-driven exploration of brain images. This includes the extraction of
the major brain networks from :term:`resting-state` data ("resting-state networks")
or movie-watching data as well as the discovery of connectionally coherent
functional modules ("connectivity-based parcellation").


References
----------

.. footbibliography::
