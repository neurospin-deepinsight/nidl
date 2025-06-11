from typing import Union, Optional
import os
import torch
from pytorch_lightning import Callback

class ModelCheckpoint(Callback):
    """ Handles model's saving periodically after a fixed amount of training steps/epochs
        By default, it only saves the last checkpoint of the training (at the epoch-level).

        TODO: monitor the quantities logged during training to enable checkpointing (see PL).
    """

    def __init__(self, 
                 dirpath: Optional[str]=None,
                 save_only_last: bool=True,
                 save_last: Union[bool, str]="link",
                 every_n_epochs: Optional[int]=None,
                 every_n_train_steps: Optional[int]=None):
        """
        Parameters
        ----------

        dirpath: Optional[str], default=None
            Directory to save the model file. By default, it will be set at runtime 
            to the location specified by `Solver`'s `root_dir` argument,
        
        save_only_last: bool, default=True
            If True, saves only the last model's checkpoint at the epoch level. 
            It is mutually exclusive with `every_n_epochs` and `every_n_train_steps`.
        
        save_last: bool or "link", default="link"
            When True, saves a last.ckpt copy whenever a checkpoint file gets saved. 
            Can be set to 'link' on a local filesystem to create a symbolic link. 
            This allows accessing the latest checkpoint in a deterministic manner.
        
        every_n_epochs:  Optional[int], default=None
            Number of epochs between checkpoints. This value must be None or non-negative. 
        
        every_n_train_steps  Optional[int], default=None
            Number of training steps between checkpoints. This value must be None or non-negative. 
        """
        if save_only_last:
            if every_n_epochs is not None or every_n_train_steps is not None:
                raise ValueError("`save_only_last` is mutually exclusive with `every_n_epochs` or `every_n_train_steps`")

        self.dirpath = dirpath
        self.save_only_last = save_only_last
        self.save_last = save_last
        self.every_n_epochs = every_n_epochs
        self.every_n_train_steps = every_n_train_steps
        self._current_checkpoint_saved = None
    
    def on_training_batch_end(self, solver):
        """ Save checkpoint on train batch end if we meet the criteria for `every_n_train_steps`
        """
        skip_batch = True

        if self.every_n_train_steps is not None:
            skip_batch = self.every_n_train_steps < 1 or (solver.global_step % self.every_n_train_steps != 0)

        if skip_batch:
            return
        
        self.save_checkpoint(solver)
    
    def on_training_epoch_end(self, solver):
        """ Save checkpoint on train batch end if we meet the criteria for `every_n_epochs`
        """
        skip_batch = True

        if self.every_n_epochs is not None:
            skip_batch = self.every_n_epochs < 1 or (solver.global_step % self.every_n_epochs != 0)
        
        if self.save_only_last:
            # Save this checkpoint no matter what
            skip_batch = False

        if skip_batch:
            return
        
        ckpt_path = self.save_checkpoint(solver)

        if self.save_only_last and self._current_checkpoint_saved is not None:
            # Remove the last checkpoint saved
            os.remove(self._current_checkpoint_saved)

        self._current_checkpoint_saved = ckpt_path


    def save_checkpoint(self, solver) -> str:
        dirpath = self.dirpath

        if dirpath is None:
            dirpath = solver.log_dir
        
        fullpath = os.path.join(
            dirpath, "checkpoints", self.format_checkpoint_name(solver.current_epoch, solver.global_step))
        
        os.makedirs(os.path.dirname(fullpath), exist_ok=True)

        torch.save(solver.state_dict(), fullpath)

        if self.save_last is True or self.save_last == "link":
            # Also save it as 'last.ckpt'
            lastpath = os.path.join(dirpath, "checkpoints", "last.ckpt")
            if os.path.isfile(lastpath):
                os.remove(lastpath)
            if self.save_last is True:
                os.makedirs(os.path.dirname(lastpath), exist_ok=True)
                torch.save(solver.state_dict(), lastpath)
            else:
                os.symlink(fullpath, lastpath)

        return fullpath


    def format_checkpoint_name(self, epoch: int, steps: int):
        return f"epoch={epoch}-step={steps}.ckpt"



        