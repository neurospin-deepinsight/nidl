##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from collections.abc import Sequence
from typing import Any, Optional

import torch


def parse_two_views_batch(
    batch: Sequence[Any],
    device: torch.device,
) -> tuple[list[torch.Tensor], Optional[torch.Tensor]]:
    """Parse a two-views batch into two views and optional labels.

    This is useful for self-supervised learning methods that require two
    augmented views of the same samples, such as SimCLR, Barlow Twins or
    y-Aware.

    Parameters
    ----------
    batch : Sequence[Any]
        Supported formats:

        - ``[X1, X2]``
        - ``([X1, X2], y)``

        where ``X1`` and ``X2`` are :class:`torch.Tensor` objects.
    device : torch.device
        The device to move the tensors to.

    Returns
    -------
    X : list[torch.Tensor]
        List containing two tensors ``[X1, X2]`` moved to the correct device.
    y : torch.Tensor or None
        Optional labels moved to the correct device.

    Raises
    ------
    ValueError
        If the batch format is not recognized or views are not tensors.
    """
    n_views = 2

    if (
        isinstance(batch, Sequence)
        and len(batch) == 2
        and isinstance(batch[0], Sequence)
        and len(batch[0]) == n_views
    ):
        X, y = batch
    elif isinstance(batch, Sequence) and len(batch) == n_views:
        X, y = batch, None
    else:
        try:
            blen = len(batch)
        except Exception:
            blen = "unknown"
        raise ValueError(
            "batch should be `[X1, X2]` or `([X1, X2], y)` where `X1` and "
            "`X2` are tensors representing two views of the same samples. "
            f"Got type={type(batch)} len={blen}."
        )

    if not (isinstance(X[0], torch.Tensor) and isinstance(X[1], torch.Tensor)):
        raise ValueError(
            "`X1` and `X2` should be torch.Tensors. "
            f"Got types: {type(X[0])}, {type(X[1])}."
        )

    X = [x.to(device, non_blocking=True) for x in X]
    if y is not None:
        if not isinstance(y, torch.Tensor):
            raise ValueError(
                f"`y` must be a torch.Tensor or None, got {type(y)}."
            )
        y = y.to(device, non_blocking=True)
    return X, y


def parse_multi_crops_batch(
    batch: Sequence[Any],
    device: torch.device,
    num_large_crops: int,
    num_local_crops: int,
) -> tuple[list[torch.Tensor], Optional[torch.Tensor]]:
    """Parse a multi-crops batch into multiple views and optional labels.

    This is useful for self-supervised learning methods that require global
    crops and local crops of the same samples, such as DINO.

    Parameters
    ----------
    batch : Sequence[Any]
        Supported formats:

        - ``[X]``
        - ``([X], y)``

        where ``[X]``is a list of torch.Tensor containing `num_large_crops`
        global views (first elements) and `num_local_crops` local
        views (last elements). `y` are eventual labels.
    device : torch.device
        The device to move the tensors to.
    num_large_crops : int
        The number of global crops in the batch.
    num_local_crops : int
        The number of local crops in the batch.

    Returns
    -------
    X : list[torch.Tensor]
        List containing the views moved to the correct device. The returned
        list length is ``num_large_crops + num_local_crops``.
    y : torch.Tensor or None
        Optional labels moved to the correct device.

    Raises
    ------
    ValueError
        If the batch format is not recognized, the number of crops does not
        match, or views are not tensors.

    Notes
    -----
    This function validates the number of crops and ensures that all crops
    share the same batch dimension, which is required by DINO-style losses
    that match views across the batch.
    """
    expected_n_crops = num_large_crops + num_local_crops
    if (
        isinstance(batch, Sequence)
        and len(batch) == 2
        and isinstance(batch[0], Sequence)
        and len(batch[0]) == expected_n_crops
    ):
        X, y = batch  # ([X], y)
    elif (
        isinstance(batch, Sequence)
        and len(batch) == expected_n_crops
        and not isinstance(batch[0], Sequence)
    ):
        X, y = batch, None
    else:
        try:
            blen = len(batch)  # type: ignore[arg-type]
        except Exception:
            blen = "unknown"
        raise ValueError(
            "batch should be `[X]` or `([X], y)` where `X` is a sequence of "
            f"{expected_n_crops} torch.Tensors (global crops first, then "
            f"local crops). Got type={type(batch)} len={blen}."
        )

    # Validate crop types and consistent batch dimension.
    crops: list[torch.Tensor] = []
    batch_size: Optional[int] = None
    for i, crop in enumerate(X):
        if not isinstance(crop, torch.Tensor):
            raise ValueError(
                "All crops must be torch.Tensors. "
                f"Crop {i} has type {type(crop)}."
            )
        if crop.ndim == 0:
            raise ValueError(
                f"Crop {i} must have a batch dimension, got ndim=0."
            )
        if batch_size is None:
            batch_size = int(crop.shape[0])
        elif int(crop.shape[0]) != batch_size:
            raise ValueError(
                "All crops must share the same first dimension (batch size). "
                f"Got crop0 batch={batch_size}, crop{i} "
                f"batch={int(crop.shape[0])}."
            )
        crops.append(crop.to(device, non_blocking=True))

    # Validate labels if present.
    y_tensor: Optional[torch.Tensor]
    if y is None:
        y_tensor = None
    else:
        if not isinstance(y, torch.Tensor):
            raise ValueError(
                f"`y` must be a torch.Tensor or None, got {type(y)}."
            )
        y_tensor = y.to(device, non_blocking=True)

    return crops, y_tensor


def gather_two_views(
    z1: torch.Tensor, z2: torch.Tensor, trainer=None, **kwargs
):
    """Gather two tensors across devices and flatten the batch dimension.

    It preserves the ordering across the two tensors and it supports gradient
    synchronization when called with `sync_grads=True`. It handles some edge
    cases, such as when using a single GPU.

    Parameters
    ----------
    z1: torch.Tensor
        Local tensor with shape ``(B, ...)``.
    z2: torch.Tensor
        Local tensor with shape ``(B, ...)`` (same order/shape as z1).
    trainer: pytorch_lightning.Trainer, default=None
        The trainer instance, used to determine the world size and whether
        distributed training is being used.
    **kwargs: dict
        Forwarded to
        :meth:`~pytorch_lightning.core.LightningModule.all_gather`.

    Returns
    -------
    z1, z2: torch.Tensor, torch.Tensor
        Gathered tensors with shape ``(B * world_size, ...)`` when running
        distributed, otherwise the input tensors.
    """
    if z1.shape != z2.shape:
        raise ValueError(
            f"z1 and z2 must have the same shape. Got {tuple(z1.shape)} and "
            f"{tuple(z2.shape)}."
        )
    if trainer is None or getattr(trainer, "world_size", 1) == 1:
        return z1, z2
    b_size = z1.shape[0]
    ws = trainer.world_size
    gathered = trainer.all_gather(torch.cat([z1, z2], dim=0), **kwargs)
    # Most Lightning strategies return (world_size, 2*batch, ...).
    if gathered.ndim < z1.ndim + 1:
        raise RuntimeError(
            f"Unexpected all_gather output shape {tuple(gathered.shape)} "
            f"for input shapes {tuple(z1.shape)} and {tuple(z2.shape)}."
        )
    # Reshape to (batch_size * world_size, *)
    z1_gathered = gathered[:, :b_size].reshape(b_size * ws, *z1.shape[1:])
    z2_gathered = gathered[:, b_size:].reshape(b_size * ws, *z2.shape[1:])
    return z1_gathered, z2_gathered


def gather_tensor(
    tensor: torch.Tensor, trainer=None, sync_grads: bool = False
):
    """
    Gather a tensor across devices and flatten the batch dimension.

    Parameters
    ----------
    tensor : torch.Tensor
        Local tensor of shape (B, ...).
    trainer : pytorch_lightning.Trainer, optional
        Trainer used to determine distributed context.
    sync_grads : bool, default=False
        Whether to synchronize gradients (set True during training).

    Returns
    -------
    torch.Tensor
        Tensor of shape (B * world_size, ...) in distributed mode,
        otherwise the input tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(tensor)}")

    if trainer is None or getattr(trainer, "world_size", 1) == 1:
        return tensor

    ws = trainer.world_size
    B = tensor.shape[0]

    gathered = trainer.all_gather(tensor, sync_grads=sync_grads)

    # Expect (world_size, B, ...)
    if gathered.ndim != tensor.ndim + 1:
        raise RuntimeError(
            f"Unexpected all_gather output shape {tuple(gathered.shape)} "
            f"for input shape {tuple(tensor.shape)}."
        )
    if gathered.shape[0] != ws or gathered.shape[1] != B:
        raise RuntimeError(
            f"Unexpected all_gather layout {tuple(gathered.shape)}. "
            f"Expected ({ws}, {B}, ...)."
        )

    return gathered.reshape(ws * B, *tensor.shape[1:])
