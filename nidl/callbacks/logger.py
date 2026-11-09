##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Custom loggers.
"""

import time
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class LiveMetricLogger(Callback):
    """
    Vertical terminal logger for PyTorch Lightning.
    
    Dynamically updates training metrics in the terminal with clear setup and
    teardown blocks for the experiment run. Uses ANSI escape codes for in-place
    updates to maintain a clean, scrolling-free display.

    Metrics are organized through a two-level grouping system that categorizes
    them by their type (step/epoch) and the training step name (val/train)
    if these keys are used in the logging name.

    Parameters
    ----------
    refresh_rate_steps : int
        How often to refresh the display (in steps).
        Default 1.
    display_width : int
        Width of the display block in characters.
        Default 50.

    Attributes
    ----------
    refresh_rate_steps : int
        Steps between display refreshes.
    display_width : int
        Width of the display block.
    _lines_written : int
        Number of lines written in current block.
    _start_time : float
        Timestamp when training started.
    """
    
    def __init__(
            self,
            refresh_rate_steps: int = 1,
            display_width: int = 50,
        ) -> None:
        super().__init__()
        self.refresh_rate_steps = refresh_rate_steps
        self.display_width = display_width
        self._lines_written = 0
        self._start_time = 0.0

    def _clear_previous_lines(self) -> None:
        """
        Move cursor up and clear lines written in the previous block.
        """
        if self._lines_written > 0:
            print("\033[F\033[K" * self._lines_written, end="")
            self._lines_written = 0

    def _render_block(
            self,
            title: str,
            metrics: dict[str, Any],
        ) -> None:
        """
        Build and print a standardized vertical grid of current metrics.

        Parameters
        ----------
        title : str
            Title to display at the top of the block.
        metrics : dict[str, Any]
            Dictionary of metric names to values.
        """
        self._clear_previous_lines()

        records = {}
        for name in sorted(metrics):
            value = metrics[name]
            dtype = "".join(
                key
                for key in ["val", "train", "step", "epoch"]
                if key in name
            )
            val_float = value.item() if hasattr(value, "item") else value
            records.setdefault(dtype, []).append(
                f" ➔ {name:<25} : {val_float:.4f}"
            )

        output_lines = [
            f"|{title:^{self.display_width}}|",
            "-" * (self.display_width + 2),
            *[
                item
                for key in sorted(records)
                for item in [*records[key], "-" * (self.display_width + 2)]
            ],
        ]
        
        print( "\n".join(output_lines))
        self._lines_written = len(output_lines)

    def _render_live(
            self,
            trainer: pl.Trainer,
            batch_idx: int,
            train: bool = True,
        ) -> None:
        """
        Dynamically refreshes metrics.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning Trainer instance.
        batch_idx : int
            Current batch index.
        train : bool
             If True, display training metrics, otherwise validation metrics.
            Default True.
        """
        if trainer.sanity_checking or batch_idx % self.refresh_rate_steps != 0:
            return

        max_epochs = trainer.max_epochs
        epoch = trainer.current_epoch + 1
        max_steps = (
            trainer.num_training_batches
            if train
            else trainer.num_val_batches[0]
        )
        title = (
            "TRAIN"
            if train
            else "VALIDATION"
        )

        self._render_block(
            title=(
                f"{title} | EPOCH {epoch}/{max_epochs} | "
                f"STEP {batch_idx + 1}/{max_steps}"
            ),
            metrics=trainer.logged_metrics
        )


    def on_fit_start(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
        ) -> None:
        """
        Display a welcome message before training begins.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning Trainer instance.
        pl_module : LightningModule
            PyTorch Lightning Module instance.
        """
        self._start_time = time.time()
        print(f"|{'FIT INITIALIZED':^{self.display_width}}|")
        print("-" * (self.display_width + 2))
        print(f" ➔ Max Epochs  : {trainer.max_epochs}")
        print(f" ➔ Devices     : {trainer.num_devices} ({trainer.device_ids})")
        print(f" ➔ Strategy    : {trainer.strategy.__class__.__name__}")
        print("-" * (self.display_width + 2))

    def on_train_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Any,
            batch: Any,
            batch_idx: int,
        ) -> None:
        """
        Dynamically refreshes metrics during training.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning Trainer instance.
        pl_module : LightningModule
            PyTorch Lightning Module instance.
        outputs : Any
            Output from the training step.
        batch : Any
            Current batch.
        batch_idx : int
            Current batch index.
        """
        self._render_live(
            trainer,
            batch_idx,
            train=True,
        )

    def on_validation_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Any,
            batch: Any,
            batch_idx: int,
        ) -> None:
        """
        Dynamically refreshes metrics during validation.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning Trainer instance.
        pl_module : LightningModule
            PyTorch Lightning Module instance.
        outputs : Any
            Output from the training step.
        batch : Any
            Current batch.
        batch_idx : int
            Current batch index.
        """
        self._render_live(
            trainer,
            batch_idx,
            train=False,
        )

    def on_fit_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
        ) -> None:
        """
        Display a summary when fitting finishes.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning Trainer instance.
        pl_module : LightningModule
            PyTorch Lightning Module instance.
        """
        self._render_block(
            title=f"SUMMARY | END OF EPOCH {trainer.current_epoch}",
            metrics=trainer.logged_metrics
        )

        elapsed_time = time.time() - self._start_time
        print(f"|{'FIT COMPLETE':^{self.display_width}}|")
        print("-" * (self.display_width + 2))
        print(f" ➔ Total Run Time : {elapsed_time:.2f} seconds")
        print(f" ➔ Final Epochs   : {trainer.current_epoch}")
        print("-" * (self.display_width + 2))
        print("\n")

        self._lines_written = 0
