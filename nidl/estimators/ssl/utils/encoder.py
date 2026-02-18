##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import copy
import logging
from typing import Any, Optional, Union

import torch.nn as nn


def build_encoder(
    encoder: Union[nn.Module, type[nn.Module]],
    encoder_kwargs: Optional[dict[str, Any]],
    deepcopy: bool = False,
) -> nn.Module:
    """Builds the encoder based on the provided configuration.

    Parameters
    ----------
    encoder : nn.Module or type[nn.Module]
        The encoder to use. It can be:
            - an already instantiated `nn.Module`, in which case it will
              be used as is.
            - a class inheriting from `nn.Module`, in which case it will
              be instantiated with the provided `encoder_kwargs`.
    encoder_kwargs : Optional[dict[str, Any]]
        The keyword arguments to use when instantiating the encoder.
        Ignored if `encoder` is already an instantiated `nn.Module`.
    deepcopy : bool, default=False
        Whether to return a deep copy of the encoder. It can be useful if the
        same encoder instance is shared across multiple SSL models.

    Returns
    -------
    nn.Module
        The instantiated encoder.
    """
    if encoder_kwargs is None:
        encoder_kwargs = {}
    if isinstance(encoder, nn.Module):
        if encoder_kwargs is not None and len(encoder_kwargs) > 0:
            logging.getLogger(__name__).warning(
                "encoder is already instantiated, ignoring 'encoder_kwargs'"
            )
        if deepcopy:  # No better way to do this currently.
            return copy.deepcopy(encoder)
    elif isinstance(encoder, type) and issubclass(encoder, nn.Module):
        encoder = encoder(**encoder_kwargs)
    else:
        raise ValueError(
            f"Encoder must be a string, a PyTorch nn.Module, or a class "
            f"inheriting from nn.Module, got {type(encoder)}"
        )
    return encoder
