##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

import typing as t

import pytorch_lightning as pl


def _isinstance_typing(value: t.Any, annot: t.Any) -> bool:
    """Minimal runtime checker for common typing annotations."""
    if annot is t.Any or annot is None:
        return True

    origin = t.get_origin(annot)
    args = t.get_args(annot)

    # Plain classes (e.g., torch.Tensor)
    if origin is None:
        try:
            return isinstance(value, annot)
        except TypeError:
            # Some typing objects aren't valid in isinstance
            return True

    # Optional / Union
    if origin is t.Union:
        return any(_isinstance_typing(value, a) for a in args)

    # Tuple[...] (fixed-length or variable-length)
    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        if len(args) == 2 and args[1] is Ellipsis:
            # tuple[T, ...]
            return all(_isinstance_typing(v, args[0]) for v in value)
        # tuple[T1, T2, ...]
        if len(value) != len(args):
            return False
        return all(_isinstance_typing(v, a) for v, a in zip(value, args))

    # List[T], Sequence[T]
    if origin in (list, t.Sequence):
        if not isinstance(value, origin if origin is list else (list, tuple)):
            return False
        if not args:
            return True
        return all(_isinstance_typing(v, args[0]) for v in value)

    # Dict[K, V]
    if origin is dict:
        if not isinstance(value, dict):
            return False
        if len(args) != 2:
            return True
        k_t, v_t = args
        return all(
            _isinstance_typing(k, k_t) and _isinstance_typing(v, v_t)
            for k, v in value.items()
        )

    # Fallback: don't block training on unsupported typing constructs
    return True


def _check_batch_against_signature(
    fn: t.Callable, batch: t.Any, arg_name: str = "batch"
) -> None:
    hints = t.get_type_hints(fn, include_extras=True)
    if arg_name not in hints:
        raise TypeError(
            f"'{fn.__qualname__}' must annotate parameter '{arg_name}'"
        )
    annot = hints[arg_name]
    if not _isinstance_typing(batch, annot):
        raise TypeError(
            f"Invalid type for '{arg_name}' in {fn.__qualname__}: "
            f"expected {annot!r}, got {type(batch).__name__}"
        )


class BatchTypingCallback(pl.Callback):
    """Check the batch format based on LightningModule step signatures.

    Raises
    ------
    TypeError
        If function parameters are not annotated or batch doesn't match.
    """

    def __init__(self, *, only_first_batch: bool = True):
        self.only_first_batch = only_first_batch

    def _maybe_check(
        self, fn: t.Callable, batch: t.Any, batch_idx: int
    ) -> None:
        if self.only_first_batch and batch_idx != 0:
            return
        _check_batch_against_signature(fn, batch, "batch")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._maybe_check(pl_module.training_step, batch, batch_idx)

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self._maybe_check(pl_module.validation_step, batch, batch_idx)

    def on_test_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self._maybe_check(pl_module.test_step, batch, batch_idx)

    def on_predict_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self._maybe_check(pl_module.predict_step, batch, batch_idx)
