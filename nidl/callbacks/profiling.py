##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import time
from typing import Any, Optional

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT


class ProfilingCallback(pl.Callback):
    """ Track the code performance.

    This callback is helpful for tracking (1) forward pass time,
    (2) backward pass time, and (3) amount of time spent waiting on the
    dataloader to generate the next batch.

    All the timings are given in seconds.

    Attributes
    ----------
    forward_time
        difference in time between
        :meth:`~ProfilingCallback.on_train_batch_start` and
        :meth:`~ProfilingCallback.on_train_batch_end`.
    val_forward_time
        difference in time between
        :meth:`~ProfilingCallback.on_validation_batch_start` and
        :meth:`~ProfilingCallback.on_validation_batch_end`.
    backward_time
        difference in time between
        :meth:`~ProfilingCallback.on_before_backwards` and
        :meth:`~ProfilingCallback.on_after_backwards`.
    step_time
        difference in time between
        :meth:`~ProfilingCallback.on_before_optimizer_step` and
        :meth:`~ProfilingCallback.on_before_zero_grad`.
        
    between_step_time
        difference in time between
        :meth:`~ProfilingCallback.on_train_batch_end` and
        :meth:`~ProfilingCallback.on_train_batch_start` (meant to capture
        time spent waiting on dataloader to generate next example).

    Notes
    -----
    Use this callback only for debugging purposes as it may slow down the
    computation.
    """

    def __init__(self):
        super().__init__()
        self.forward_time = 0.0
        self.val_forward_time = 0.0
        self.step_time = 0.0
        self.between_step_time = 0.0

    def on_train_batch_start(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            batch: Any,
            batch_idx: int) -> None:
        self.batch_start = time.perf_counter()
        if hasattr(self, "batch_end"):
            self.between_step_time = time.perf_counter() - self.batch_end
            pl_module.log(
                "train/between_step_time_seconds",
                self.between_step_time,
                on_step=True,
                on_epoch=False,
                rank_zero_only=True
            )

    def on_train_batch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int) -> None:
        self.forward_time = time.perf_counter() - self.batch_start
        pl_module.log(
            "train/forward_time_seconds",
            self.forward_time,
            on_step=True,
            on_epoch=False,
            rank_zero_only=True
        )
        self.batch_end = time.perf_counter()

    def on_before_backward(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            loss: Any) -> None:
        self.backward_start = time.perf_counter()

    def on_after_backward(
            self,
            trainer: Trainer,
            pl_module: LightningModule) -> None:
        self.backward_time = time.perf_counter() - self.backward_start
        pl_module.log(
            "train/backward_time_seconds",
            self.backward_time,
            on_step=True,
            on_epoch=False,
            rank_zero_only=True
        )

    def on_validation_batch_start(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            batch: Any,
            batch_idx: int,
            dataloader_idx: Optional[int] = 0) -> None:
        self.val_batch_start = time.perf_counter()

    def on_validation_batch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx=0) -> None:
        self.val_forward_time = time.perf_counter() - self.val_batch_start
        pl_module.log(
            "validation/forward_time_seconds",
            self.val_forward_time,
            on_step=True,
            on_epoch=False,
            rank_zero_only=True
        )

    def on_before_optimizer_step(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            optimizer: Any) -> None:
        self.step_start = time.perf_counter()

    def on_before_zero_grad(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            optimizer: Any) -> None:
        if hasattr(self, "step_start"):
            self.step_time = time.perf_counter() - self.step_start
            pl_module.log(
                "train/step_time_seconds",
                self.step_time,
                on_step=True,
                on_epoch=False,
                rank_zero_only=True
            )
