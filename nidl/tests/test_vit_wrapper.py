##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import unittest
from types import SimpleNamespace

from nidl.estimators.ssl.utils.vit_wrapper import ViTWrapperMixin


class _Wrapper(ViTWrapperMixin):
    """ Minimal wrapper: stores a ViT-like object in ``self.vit``.
    """
    _REQUIRED = ("embed_dim", "has_class_token")

    def __init__(self, vit):
        self.vit = vit


def _fake_vit(**kwargs):
    """ ViT-like object exposing only the given attributes.
    """
    return SimpleNamespace(**kwargs)


class TestViTWrapperMixin(unittest.TestCase):
    """ Test the attributes shared by the I-JEPA and NeuroJEPA encoder
    wrappers.
    """
    def test_should_list_missing_required_attributes(self):
        """ Required attributes absent from the ViT are returned.
        """
        vit = _fake_vit(embed_dim=8)
        wrapper = _Wrapper(vit)

        missing = wrapper._is_vit_like(vit)

        self.assertEqual(missing, ["has_class_token"])

    def test_should_expose_interface(self):
        """ The shared attributes are read from the wrapped ViT.
        """
        block = SimpleNamespace(attn=SimpleNamespace(num_heads=3))
        vit = _fake_vit(
            embed_dim=8, blocks=[block], has_class_token=True,
            num_reg_tokens=4
        )

        wrapper = _Wrapper(vit)

        self.assertEqual(wrapper.embed_dim, vit.embed_dim)
        self.assertEqual(wrapper.num_heads, block.attn.num_heads)
        self.assertEqual(wrapper.has_class_token, vit.has_class_token)
        self.assertEqual(wrapper.num_reg_tokens, vit.num_reg_tokens)

    def test_should_count_class_and_register_tokens_as_prefix(self):
        """ Without ViT bookkeeping, prefix = class token + register tokens.
        """
        wrapper = _Wrapper(_fake_vit(has_class_token=True, num_reg_tokens=2))

        n_prefix = wrapper.num_prefix_tokens

        self.assertEqual(n_prefix, 3)

    def test_should_prefer_num_prefix_tokens_of_vit(self):
        """ The ViT's own ``num_prefix_tokens`` is used when it exists.
        """
        wrapper = _Wrapper(_fake_vit(
            has_class_token=True, num_reg_tokens=2, num_prefix_tokens=5
        ))

        n_prefix = wrapper.num_prefix_tokens

        self.assertEqual(n_prefix, 5)


if __name__ == "__main__":
    unittest.main()
