##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import pytorch_lightning as pl
from typex import check_type


class BatchTypingCallback(pl.Callback):
    """ This callback allows you to check the batch format based on the
    method signature.

    Raises
    ------
    TypeError
        If function parameters are not annotated.
    TraitError
        If the input value have incorrect type.
    NotImplementedError
        If a type is not handled by the code.
    """
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        check_type(batch, pl_module.training_step, "batch")

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx,
                                  dataloader_idx=0):
        check_type(batch, pl_module.validation_step, "batch")

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx,
                            dataloader_idx=0):
        check_type(batch, pl_module.test_step, "batch")

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx,
                               dataloader_idx=0):
        check_type(batch, pl_module.predict_step, "batch")
