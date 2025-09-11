##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from typing import Any

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT

from nidl.utils import Bunch


class CachingCallback(pl.Callback):
    """ Cache embeddings.

    Attributes
    ----------
    returned_objects
        a :class:`~nidl.utils.Bunch` object containing the the different
        objects returned after the validation in `validation` and `last_epoch`.

    Notes
    -----
    The caching is performed during the
    :meth:`~CachingCallback.on_validation_batch_end` and
    :meth:`~CachingCallback.on_test_epoch_end` events.
    """
    def __init__(self):
        super().__init__()
        self.returned_objects_ = Bunch()
        self.validation_cache = []

    @property
    def returned_objects(self):
	    return self.returned_objects_

    def on_validation_epoch_start(
            self,
            trainer: Trainer,
            pl_module: LightningModule) -> None:
        self.validation_cache.clear()

    def on_validation_batch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0) -> None:
        if trainer.sanity_checking:
            return
        self.validation_cache.append(outputs)

    def on_validation_epoch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule) -> None:
        self.returned_objects_["last_epoch"] = pl_module.current_epoch
        self.returned_objects_["validation"] = self.validation_cache
        pl_module.cache = self.returned_objects_

    def load_state_dict(self, state_dict):
        self.returned_objects_.update(state_dict)

    def state_dict(self):
        return self.returned_objects.copy()
