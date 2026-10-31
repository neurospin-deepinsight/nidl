##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional, Union

import numpy as np
import torch
from torch.utils.data import default_collate


def inspect_batch(x, name="batch", max_items=4) -> str:
    """
    Return a human-readable string describing the nested structure
    of a batch (types, lengths, tensor shapes, dtypes).

    Parameters
    ----------
    x : Any
        Object to inspect.
    name : str
        Root name.
    max_items : int
        Max number of sequence elements to display per level.
    """
    lines = []

    def _inspect(obj, prefix, indent=0):
        pad = " " * indent
        t = type(obj)

        if isinstance(obj, torch.Tensor):
            lines.append(
                f"{pad}{prefix}: torch.Tensor "
                f"shape={tuple(obj.shape)} dtype={obj.dtype}"
            )

        elif isinstance(obj, np.ndarray):
            lines.append(
                f"{pad}{prefix}: np.ndarray "
                f"shape={obj.shape} dtype={obj.dtype}"
            )

        elif isinstance(obj, Mapping):
            keys = list(obj.keys())
            lines.append(f"{pad}{prefix}: dict keys={keys}")
            for k in keys:
                _inspect(obj[k], f"{prefix}['{k}']", indent + 2)

        elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
            length = len(obj)
            lines.append(f"{pad}{prefix}: {t.__name__} len={length}")
            for i, v in enumerate(obj[:max_items]):
                _inspect(v, f"{prefix}[{i}]", indent + 2)
            if length > max_items:
                lines.append(
                    f"{pad}  ... ({length - max_items} more elements)"
                )

        else:
            lines.append(f"{pad}{prefix}: {t}")

    _inspect(x, name)
    return "\n".join(lines)


def parse_xy_batch(
    batch: tuple[torch.Tensor, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse a (X, y) labeled batch into a pair of torch Tensors.

    This is useful for supervised models requiring labels.

    Parameters
    ----------
    batch : tuple[torch.Tensor, torch.Tensor]
        Supported format is ``(X, y)`` where ``X``and ``y``are
        torch.Tensor with same first (batch) dimension.
    device : torch.device
        The device to move the tensors to.

    Returns
    -------
    X : torch.Tensor
        Input data moved to the correct device.
    y : torch.Tensor
        Labels moved to the correct device.

    Raises
    ------
    ValueError
        If the batch format is not recognized or labels are not present.
    """
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError(
            "Batch must be a tuple/list of (X, y) tensors. Got\n"
            + inspect_batch(batch)
        )

    X, y = batch

    if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError(
            "Both X and y must be torch.Tensor. Got\n" + inspect_batch(batch)
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Batch size mismatch: X has {X.shape[0]} samples but y has "
            f"{y.shape[0]}\n" + inspect_batch(batch)
        )

    X = X.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    return X, y


def parse_x_or_xy_batch(
    batch: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Parse a X or (X, y) batch into a pair of torch Tensors.

    This is useful for supervised and unsupervised models.

    Parameters
    ----------
    batch : torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        Supported format is ``X`` and ``(X, y)`` where ``X``and ``y``are
        torch.Tensor with same first (batch) dimension.
    device : torch.device
        The device to move the tensors to.

    Returns
    -------
    X : torch.Tensor
        Input data moved to the correct device.
    y : torch.Tensor or None
        Optional labels moved to the correct device.

    Raises
    ------
    ValueError
        If the batch format is not recognized.
    """
    # Case 1: unlabeled batch
    if isinstance(batch, torch.Tensor):
        X = batch.to(device, non_blocking=True)
        return X, None

    # Case 2: labeled batch
    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            X, y = parse_xy_batch(batch, device)
            return X, y
        elif len(batch) == 1:
            X = batch[0]
            if not isinstance(X, torch.Tensor):
                raise TypeError(
                    "Batch element must be a torch.Tensor. Got\n"
                    + inspect_batch(batch)
                )
            return X.to(device, non_blocking=True), None

    raise TypeError(
        "Unsupported batch format. Expected X or (X, y) where both are "
        "tensors. Got\n" + inspect_batch(batch)
    )


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

        where ``X1`` and ``X2`` are :class:`torch.Tensor` with same shape.
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
        If the batch format is not recognized or views are not tensors or with
        different shapes.
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
        raise ValueError(
            "batch should be `[X1, X2]` or `([X1, X2], y)` where `X1` and "
            "`X2` are tensors representing two views of the same samples. "
            "Got\n" + inspect_batch(batch)
        )

    if not (isinstance(X[0], torch.Tensor) and isinstance(X[1], torch.Tensor)):
        raise TypeError(
            "`X1` and `X2` should be torch.Tensors. "
            "Got\n" + inspect_batch(batch)
        )

    if X[0].shape != X[1].shape:
        raise ValueError(
            "`X1` and `X2` should have same shape. "
            "Got\n" + inspect_batch(batch)
        )

    X = [x.to(device, non_blocking=True) for x in X]
    if y is not None:
        if not isinstance(y, torch.Tensor):
            raise ValueError(
                "`y` must be a torch.Tensor or None. "
                "Got\n" + inspect_batch(batch)
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
        match, views are not tensors or global (resp. local) crops shape does
        not match.

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
        raise ValueError(
            "batch should be `[X]` or `([X], y)` where `X` is a sequence of "
            f"{expected_n_crops} torch.Tensors (global crops first, then "
            f"local crops). "
            "Got\n" + inspect_batch(batch)
        )

    # Validate crop types and consistent shape across crops (global or local).
    crops: list[torch.Tensor] = []
    ref_global: Optional[torch.Tensor] = None
    ref_local: Optional[torch.Tensor] = None
    for i, crop in enumerate(X):
        if not isinstance(crop, torch.Tensor) or crop.ndim == 0:
            raise ValueError(
                "All crops must be torch.Tensors with a batch dimension. "
                "Got\n" + inspect_batch(batch)
            )
        ref = ref_global if i < num_large_crops else ref_local
        if ref is None:
            if i < num_large_crops:
                ref_global = crop
            else:
                ref_local = crop
        elif crop.shape != ref.shape:
            group = "global" if i < num_large_crops else "local"
            raise ValueError(
                f"All {group} crops must have the same shape. "
                "Got\n" + inspect_batch(batch)
            )
        crops.append(crop.to(device, non_blocking=True))

    # Validate labels if present.
    y_tensor: Optional[torch.Tensor]
    if y is None:
        y_tensor = None
    else:
        if not isinstance(y, torch.Tensor):
            raise ValueError(
                "`y` must be a torch.Tensor or None."
                "Got\n" + inspect_batch(batch)
            )
        y_tensor = y.to(device, non_blocking=True)

    return crops, y_tensor


def collate_multimodal(
    batch: Sequence[Any],
    modalities: Optional[Sequence[str]] = None,
) -> Union[dict[str, Any], tuple[dict[str, Any], Any]]:
    """Collate samples with missing modalities into a multimodal batch.

    Each sample is a dict ``{modality: data or None}``, or a ``(dict, y)``
    pair, where ``data`` is a scan or a list of augmented views of a scan.
    The scans of the present modalities are concatenated over the batch,
    along with their sample index and modality slot.

    Parameters
    ----------
    batch : sequence
        The samples of the batch.
    modalities : sequence of str or None, default=None
        The modalities, in the order of their slots. By default, the keys of
        the first sample.

    Returns
    -------
    X : dict
        Dictionary containing:

        - "views" (list of tensors, one per view) or "x" (tensor): the scans
          of the present modalities, of shape ``(n_scans, C, *)``;
        - "sample_idx": the sample index of each scan, of shape
          ``(n_scans,)``;
        - "mod_idx": the modality slot of each scan, of shape ``(n_scans,)``;
        - "n_samples": the number of samples in the batch.
    y : torch.Tensor, optional
        The collated targets, only if the samples are ``(dict, y)`` pairs.

    Raises
    ------
    TypeError
        If a sample is not a dict or a ``(dict, y)`` pair.
    ValueError
        If a sample has no modality or an unknown one, or if the scans do not
        hold the same number of views.
    """
    with_targets = isinstance(batch[0], (tuple, list))
    samples = [sample[0] for sample in batch] if with_targets else batch
    if modalities is None:
        modalities = list(samples[0])
    slots = {modality: idx for idx, modality in enumerate(modalities)}
    views, sample_idx, mod_idx = [], [], []
    multi_views, n_views = None, None
    for i, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise TypeError(
                "Each sample should be a dict {modality: data or None} or a "
                "(dict, y) pair. Got\n" + inspect_batch(sample, name="sample")
            )
        unknown = set(sample) - set(slots)
        if unknown:
            raise ValueError(
                f"Unknown modalities {sorted(unknown)} in sample {i}, "
                f"expected among {list(modalities)}."
            )
        n_present = 0
        for modality in modalities:
            data = sample.get(modality)
            if data is None:
                continue
            is_multi = isinstance(data, (tuple, list))
            data = list(data) if is_multi else [data]
            if multi_views is None:
                multi_views, n_views = is_multi, len(data)
            if is_multi != multi_views or len(data) != n_views:
                raise ValueError(
                    "All the scans should hold the same number of views "
                    "(a single scan, or a list of views). Got a different "
                    f"one for {modality!r} in sample {i}."
                )
            views.append([torch.as_tensor(view) for view in data])
            sample_idx.append(i)
            mod_idx.append(slots[modality])
            n_present += 1
        if n_present == 0:
            raise ValueError(f"Sample {i} of the batch has no modality.")

    stacked = [torch.stack(scans) for scans in zip(*views)]
    X = {"views": stacked} if multi_views else {"x": stacked[0]}
    X["sample_idx"] = torch.tensor(sample_idx, dtype=torch.long)
    X["mod_idx"] = torch.tensor(mod_idx, dtype=torch.long)
    X["n_samples"] = len(samples)
    if with_targets:
        return X, default_collate([sample[1] for sample in batch])
    return X


def parse_multimodal_batch(
    batch: Any,
    device: torch.device,
    n_views: Optional[int] = None,
) -> tuple[dict[str, Any], Optional[torch.Tensor]]:
    """Parse a multimodal batch with missing modalities into a dict and
    optional labels.

    This is useful for multimodal methods handling missing modalities, such
    as CoMM. Such batches are built by :func:`collate_multimodal`.

    Parameters
    ----------
    batch : dict or tuple of (dict, torch.Tensor)
        Supported formats:

        - ``X``
        - ``(X, y)``

        where ``X`` is a dict as returned by :func:`collate_multimodal`.
    device : torch.device
        The device to move the tensors to.
    n_views : int or None, default=None
        Number of views expected under the "views" key of ``X``. If None, a
        single tensor is expected under the "x" key.

    Returns
    -------
    X : dict
        Dict with keys "views" (or "x"), "sample_idx", "mod_idx" and
        "n_samples", with the tensors moved to the correct device.
    y : torch.Tensor or None
        Optional labels moved to the correct device.

    Raises
    ------
    TypeError
        If the scans are not tensors.
    ValueError
        If the batch format is not recognized.
    """
    if isinstance(batch, Mapping):
        X, y = batch, None
    elif (
        isinstance(batch, Sequence)
        and len(batch) == 2
        and isinstance(batch[0], Mapping)
    ):
        X, y = batch
    else:
        raise ValueError(
            "batch should be `X` or `(X, y)` where `X` is a dict as returned "
            "by `collate_multimodal`. Got\n" + inspect_batch(batch)
        )

    key = "x" if n_views is None else "views"
    missing = {key, "sample_idx", "mod_idx", "n_samples"} - set(X)
    if missing:
        hint = ""
        if key == "x" and "views" in X:
            hint = " (expected a single scan per modality, not views)"
        elif key == "views" and "x" in X:
            hint = f" (expected {n_views} views per modality)"
        raise ValueError(
            f"batch misses the key(s) {sorted(missing)}{hint}. Got\n"
            + inspect_batch(batch)
        )

    if n_views is None:
        if not isinstance(X["x"], torch.Tensor):
            raise TypeError(
                "`x` should be a torch.Tensor. Got\n" + inspect_batch(batch)
            )
        data = X["x"].to(device, non_blocking=True)
        n_scans = len(data)
    else:
        views = X["views"]
        if (
            not isinstance(views, Sequence)
            or len(views) != n_views
            or not all(isinstance(view, torch.Tensor) for view in views)
        ):
            raise TypeError(
                f"`views` should be a list of {n_views} torch.Tensors. Got\n"
                + inspect_batch(batch)
            )
        if any(view.shape != views[0].shape for view in views):
            raise ValueError(
                "All the views should have the same shape. Got\n"
                + inspect_batch(batch)
            )
        data = [view.to(device, non_blocking=True) for view in views]
        n_scans = len(data[0])

    out = {key: data, "n_samples": int(X["n_samples"])}
    for name in ("sample_idx", "mod_idx"):
        idx = X[name]
        if not isinstance(idx, torch.Tensor) or tuple(idx.shape) != (
            n_scans,
        ):
            raise ValueError(
                f"`{name}` should be a tensor of shape ({n_scans},). Got\n"
                + inspect_batch(batch)
            )
        out[name] = idx.to(device, non_blocking=True).long()

    if y is not None:
        if not isinstance(y, torch.Tensor):
            raise ValueError(
                "`y` must be a torch.Tensor or None. Got\n"
                + inspect_batch(batch)
            )
        y = y.to(device, non_blocking=True)
    return out, y


def gather_two_views(
    z1: torch.Tensor, z2: torch.Tensor, module=None, **kwargs
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
    module: pytorch_lightning.LightningModule, default=None
        The module instance, used to determine the world size and whether
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
    trainer = module.trainer if module is not None else None
    if trainer is None or getattr(trainer, "world_size", 1) == 1:
        return z1, z2
    b_size = z1.shape[0]
    ws = trainer.world_size
    gathered = module.all_gather(torch.cat([z1, z2], dim=0), **kwargs)
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


def gather_tensor(tensor: torch.Tensor, module=None, sync_grads: bool = False):
    """
    Gather a tensor across devices and flatten the batch dimension.

    Parameters
    ----------
    tensor : torch.Tensor
        Local tensor of shape (B, ...).
    module : pytorch_lightning.LightningModule, optional
        Module used to determine distributed context.
    sync_grads : bool, default=False
        Whether to synchronize gradients (set True during training).

    Returns
    -------
    torch.Tensor
        Tensor of shape (B * world_size, ...) in distributed mode,
        otherwise the input tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    trainer = module.trainer if module is not None else None

    if trainer is None or getattr(trainer, "world_size", 1) == 1:
        return tensor

    ws = trainer.world_size
    B = tensor.shape[0]

    gathered = module.all_gather(tensor, sync_grads=sync_grads)

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
