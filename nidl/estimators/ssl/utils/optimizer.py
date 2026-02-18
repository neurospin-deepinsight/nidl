##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import logging
from typing import Any, Optional, Union

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import LRSchedulerPLType
from torch.optim import Optimizer

from nidl.utils import LinearWarmupCosineAnnealingLR

_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamW": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

_LR_SCHEDULERS = ["none", "warmup_cosine"]


def configure_ssl_optimizers(
    trainer: Trainer,
    optim_params: list[dict],
    optimizer: Optimizer,
    optimizer_kwargs: Optional[dict[str, Any]],
    learning_rate: float,
    weight_decay: float,
    exclude_bias_and_norm_wd: bool,
    lr_scheduler: Optional[Union[str, LRSchedulerPLType]],
    lr_scheduler_kwargs: Optional[dict[str, Any]],
):
    """Initialize the optimizer and learning rate scheduler in SSL."""
    if exclude_bias_and_norm_wd:
        optim_params = remove_bias_and_norm_from_weight_decay(optim_params)

    if isinstance(optimizer, str):
        if optimizer not in _OPTIMIZERS:
            raise ValueError(
                f"Optimizer '{optimizer}' is not implemented. "
                f"Please use one of the available optimizers: "
                f"{', '.join(_OPTIMIZERS.keys())}"
            )
        optimizer = _OPTIMIZERS[optimizer](
            params=optim_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            **(optimizer_kwargs or {}),
        )
    elif isinstance(optimizer, Optimizer):
        if len(optimizer_kwargs) > 0:
            logging.getLogger(__name__).warning(
                "optimizer is already instantiated, ignoring "
                "'optimizer_kwargs'"
            )
        optimizer = optimizer
    else:
        raise ValueError(
            "Optimizer must be a string or a PyTorch Optimizer, got "
            f"{type(optimizer)}"
        )
    if lr_scheduler is None:
        return optimizer

    if isinstance(lr_scheduler, str):
        if lr_scheduler_kwargs is None:
            lr_scheduler_kwargs = {}
        if lr_scheduler in _LR_SCHEDULERS:
            if lr_scheduler == "warmup_cosine":
                warmup_epochs = int(lr_scheduler_kwargs["warmup_epochs"])
                interval = str(lr_scheduler_kwargs["interval"])
                warmup_start_lr = float(lr_scheduler_kwargs["warmup_start_lr"])
                min_lr = float(lr_scheduler_kwargs["min_lr"])

                if interval not in {"step", "epoch"}:
                    raise ValueError(f"Unknown interval: {interval}")

                if trainer.max_epochs in (None, -1, 0):
                    # Potentially infinite training loop
                    raise ValueError(
                        "`max_epoch` must be set in your Trainer to use warmup"
                        " in the LR scheduler."
                    )
                max_epochs = trainer.max_epochs
                max_warmup_steps = (
                    warmup_epochs
                    * (trainer.estimated_stepping_batches / max_epochs)
                    if interval == "step"
                    else warmup_epochs
                )
                max_scheduler_steps = (
                    trainer.estimated_stepping_batches
                    if interval == "step"
                    else max_epochs
                )
                scheduler = {
                    "scheduler": LinearWarmupCosineAnnealingLR(
                        optimizer,
                        warmup_epochs=max_warmup_steps,
                        max_epochs=max_scheduler_steps,
                        warmup_start_lr=warmup_start_lr
                        if warmup_epochs > 0
                        else learning_rate,
                        eta_min=min_lr,
                    ),
                    "interval": interval,
                    "frequency": 1,
                }
            elif lr_scheduler == "none":
                return optimizer
        else:
            raise ValueError(f"Unknown `lr_scheduler`: {lr_scheduler}")
    elif isinstance(lr_scheduler, LRSchedulerPLType):
        if len(lr_scheduler_kwargs) > 0:
            logging.getLogger(__name__).warning(
                "lr_scheduler is already instantiated, ignoring "
                "'lr_scheduler_kwargs'"
            )
        scheduler = lr_scheduler
    else:
        raise ValueError(
            f"Unknown type for `lr_scheduler`: {type(lr_scheduler)}"
        )

    return [optimizer], [scheduler]


def remove_bias_and_norm_from_weight_decay(parameter_groups: list[dict]):
    out = []
    for group in parameter_groups:
        # parameters with weight decay
        decay_group = {k: v for k, v in group.items() if k != "params"}

        # parameters without weight decay
        no_decay_group = {k: v for k, v in group.items() if k != "params"}
        no_decay_group["weight_decay"] = 0
        group_name = group.get("name", None)
        if group_name:
            no_decay_group["name"] = group_name + "_no_decay"

        # split parameters into the two lists
        decay_params = []
        no_decay_params = []
        for param in group["params"]:
            if param.ndim <= 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # add groups back
        if decay_params:
            decay_group["params"] = decay_params
            out.append(decay_group)
        if no_decay_params:
            no_decay_group["params"] = no_decay_params
            out.append(no_decay_group)
    return out
