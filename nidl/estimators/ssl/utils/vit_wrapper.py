##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

from typing import Any


class ViTWrapperMixin:
    """
    Common building block for wrappers that adapt a Vision Transformer (ViT)
    to a specific purpose.

    Many methods need the same ViT backbone, but each one expects a slightly
    different API from it: a self-supervised estimator may need masked
    tokens, while a probing tool may only need patch features. A wrapper
    bridges this gap by exposing the API the method expects on top of a
    timm-like ViT. This mixin gathers the code that every such wrapper
    needs, so that it is written once:

    - a shallow check that the wrapped ViT exposes the expected interface;
    - read-only access to the ViT's main hyperparameters: ``embed_dim``,
      ``num_heads``, ``has_class_token``, ``num_reg_tokens`` and
      ``num_prefix_tokens``.

    For example, the encoder wrappers of I-JEPA
    (:class:`~nidl.estimators.ssl.ijepa.IJEPAVisionTransformer`) and
    NeuroJEPA (:class:`~nidl.estimators.ssl.neurojepa.NeuroJEPAEncoderWrapper`)
    are built on it.

    Notes
    -----
    To build a new wrapper, inherit from this mixin (before ``nn.Module``),
    store the wrapped ViT in ``self.vit``, and list the attributes or
    methods the ViT must expose in the ``_REQUIRED`` class attribute. Then
    call ``self._is_vit_like(vit)`` in ``__init__`` to get the missing ones.
    """

    _REQUIRED: tuple[str, ...] = ()

    def _is_vit_like(self, vit: Any) -> list[str]:
        """
        Check whether a ``nn.Module`` follows the timm-like ViT interface.

        Parameters
        ----------
        vit : nn.Module
            Module to validate.

        Returns
        -------
        list[str]
            Names of the attributes or methods of ``_REQUIRED`` that are
            missing from ``vit``. An empty list indicates that the module
            matches the expected interface.

        Notes
        -----
        This is a shallow interface check. It verifies the presence of
        required members, but does not validate their semantics,
        signatures, or runtime behavior.
        """
        return [a for a in self._REQUIRED if not hasattr(vit, a)]

    @property
    def embed_dim(self) -> int:
        """Token embedding dimension of the wrapped encoder."""
        return self.vit.embed_dim

    @property
    def num_heads(self) -> int:
        """Number of attention heads, read from the first block."""
        return self.vit.blocks[0].attn.num_heads

    @property
    def has_class_token(self) -> bool:
        """Whether the wrapped encoder prepends a class token."""
        return self.vit.has_class_token

    @property
    def num_reg_tokens(self) -> int:
        """Number of register tokens of the wrapped encoder."""
        return self.vit.num_reg_tokens

    @property
    def num_prefix_tokens(self) -> int:
        """
        Number of tokens preceding the patch tokens.

        The wrapped encoder's own ``num_prefix_tokens`` is used if it exists.
        Otherwise, the class token and the register tokens are counted.
        """
        if hasattr(self.vit, "num_prefix_tokens"):
            return self.vit.num_prefix_tokens
        return int(self.has_class_token) + self.num_reg_tokens
