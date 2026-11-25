##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Load weights that are published by their authors in another format.

Some weights cannot be published again in the nidl format, for instance
because their license forbids derivatives. Such a model is still published
like any other, with a ``config.yaml`` in the nidl format (see
:class:`~nidl.estimators.BaseEstimator`), but without weights: its
``weights`` entry points to the repository of the authors, and names a
converter of this module:

.. code-block:: yaml

    nidl_version: 0.0.3
    estimator:
      _target_: nidl.estimators.ssl.neurojepa.NeuroJEPA
      ...
    weights:
      repo_id: NYUMedML/Neuro-JEPA
      revision: f55b154091aa7ac5ca23632918e216cd92d9f2cb
      filename: model.safetensors
      converter: NeuroJEPAWeights
      missing_ok: [predictor.]

The ``weights`` entry contains:

- ``repo_id``: the Hugging Face repository of the authors;
- ``revision``: the commit of this repository validated with nidl. It is
  required, so that the weights cannot change without notice;
- ``filename`` (optional, default `model.safetensors`): the safetensors
  file holding the weights;
- ``converter`` (optional): the name of a subclass of
  :class:`ExternalWeights` defined in this module;
- ``missing_ok`` (optional): prefixes of the parameters of the estimator
  that the weights do not include, which keep their initial values.

When the estimator is loaded with
:meth:`~nidl.utils.hub.HubMixin.from_pretrained`, the weights are
downloaded from the repository of the authors, renamed by the converter and
loaded strictly in the estimator built from the configuration.
"""

from __future__ import annotations

import abc
import sys
from collections.abc import Mapping

import torch


def get_converter(name: str) -> ExternalWeights:
    """Return the converter of weights named in a configuration.

    Parameters
    ----------
    name : str
        The name of a subclass of :class:`ExternalWeights` defined in this
        module.

    Returns
    -------
    converter : ExternalWeights
        An instance of this subclass.

    Raises
    ------
    ValueError
        If `name` is not a converter of this module.
    """
    converter = getattr(sys.modules[__name__], str(name), None)
    if (
        str(name).startswith("_")
        or not isinstance(converter, type)
        or not issubclass(converter, ExternalWeights)
        or converter is ExternalWeights
    ):
        raise ValueError(f"Unknown converter '{name}'.")
    return converter()


class ExternalWeights(abc.ABC):
    """Contract of the converters of weights published in another format.

    A converter only renames the weights: :meth:`convert` takes the state
    dictionary published by the authors and returns the one of the nidl
    estimator described by the configuration. It must not depend on
    anything else, so that the configuration alone describes the estimator.

    To publish such a model:

    1) implement a subclass of :class:`ExternalWeights` in this module;
    2) write the ``config.yaml`` of the estimator, i.e., its configuration
       (see :func:`~nidl.utils.config.get_config`), with a ``weights`` entry
       naming the repository, the revision and the converter;
    3) check that the estimator loaded with
       :meth:`~nidl.utils.hub.HubMixin.from_pretrained` returns the
       same features as the official code;
    4) publish the ``config.yaml`` and the ``preprocessor_config.yaml`` of
       its preprocessing, with a model card, to the `nidl-neurospin
       <https://huggingface.co/nidl-neurospin>`_ organization.
    """

    @abc.abstractmethod
    def convert(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Rename the published weights after the modules of nidl.

        Parameters
        ----------
        state_dict : mapping of str to torch.Tensor
            The weights, as published by their authors.

        Returns
        -------
        state_dict : dict of str to torch.Tensor
            The weights, named as the parameters of the nidl estimator.
            Weights that the converter does not know keep their name, so
            that they are reported as unexpected when loading.
        """


class NeuroJEPAWeights(ExternalWeights):
    """Official Neuro-JEPA weights [1]_, gated, under CC-BY-NC-ND 4.0.

    Only the encoder is published, with the prefix `module.backbone.`. It is
    loaded in both the `context_encoder` and the `target_encoder` of
    :class:`~nidl.estimators.ssl.NeuroJEPA`, the `predictor` staying random.

    References
    ----------
    .. [1] Huang et al., "Learning Sparse Latent Predictive Foundation
           Model for Multimodal Neuroimaging", arXiv:2606.14957.
           https://github.com/NYUMedML/Neuro-JEPA
    """

    prefix = "module.backbone."

    def convert(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        converted = {}
        for name, value in state_dict.items():
            if not name.startswith(self.prefix):
                converted[name] = value
                continue
            name = name[len(self.prefix) :]
            converted[f"context_encoder.vit.{name}"] = value
            converted[f"target_encoder.vit.{name}"] = value
        return converted
