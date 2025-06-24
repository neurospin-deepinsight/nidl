from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.strategies import Strategy
from pytorch_lightning.loggers import Logger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.profilers import Profiler
from pytorch_lightning.plugins import _PLUGIN_INPUT
from pytorch_lightning.trainer.connectors.accelerator_connector import _LITERAL_WARN, _PRECISION_INPUT
from lightning_fabric.utilities.types import _PATH
from typing import Union, Optional, Iterable, Any
from datetime import timedelta
from abc import ABC
import torch
from nidl.utils.validation import check_is_fitted

class BaseEstimator(ABC, LightningModule):
    """
    Base class for all estimators in the NIDL framework designed for scalability.
    Inherits from PyTorch Lightning's LightningModule.
    This class provides a common interface for training, validation, and prediction in 
    a distributed setting (multi-node multi-GPU).

    It provides you (already implemented):
     - 'log_dict' method for logging a dictionary of metrics
     - 'log' method for logging a metric
     - 'fit' method for fitting the model to data
     - 'predict' method for making predictions on new data
     - 'predict_step' method for defining the prediction step
     - 'training_step' method for defining the training step
     - 'validation_step' method for defining the validation step
     - 'test_step' method for defining the test step
     - 'forward' method for defining the forward pass of the model

    """

    def __init__(
            self,
            accelerator: Union[str, Accelerator] = "auto",
            strategy: Union[str, Strategy] = "auto",
            devices: Union[list[int], str, int] = "auto",
            num_nodes: int = 1,
            precision: Optional[_PRECISION_INPUT] = None,
            logger: Optional[Union[Logger, Iterable[Logger], bool]] = None,
            callbacks: Optional[Union[list[Callback], Callback]] = None,
            fast_dev_run: Union[int, bool] = False,
            max_epochs: Optional[int] = None,
            min_epochs: Optional[int] = None,
            max_steps: int = -1,
            min_steps: Optional[int] = None,
            max_time: Optional[Union[str, timedelta, dict[str, int]]] = None,
            limit_train_batches: Optional[Union[int, float]] = None,
            limit_val_batches: Optional[Union[int, float]] = None,
            limit_test_batches: Optional[Union[int, float]] = None,
            limit_predict_batches: Optional[Union[int, float]] = None,
            overfit_batches: Union[int, float] = 0.0,
            val_check_interval: Optional[Union[int, float]] = None,
            check_val_every_n_epoch: Optional[int] = 1,
            num_sanity_val_steps: Optional[int] = None,
            log_every_n_steps: Optional[int] = None,
            enable_checkpointing: Optional[bool] = None,
            enable_progress_bar: Optional[bool] = None,
            enable_model_summary: Optional[bool] = None,
            accumulate_grad_batches: int = 1,
            gradient_clip_val: Optional[Union[int, float]] = None,
            gradient_clip_algorithm: Optional[str] = None,
            deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
            benchmark: Optional[bool] = None,
            inference_mode: bool = True,
            use_distributed_sampler: bool = True,
            profiler: Optional[Union[Profiler, str]] = None,
            detect_anomaly: bool = False,
            barebones: bool = False,
            plugins: Optional[Union[_PLUGIN_INPUT, list[_PLUGIN_INPUT]]] = None,
            sync_batchnorm: bool = False,
            reload_dataloaders_every_n_epochs: int = 0,
            default_root_dir: Optional[_PATH] = None
            ):
        """
            Parameters
            ----------
            accelerator: Supports passing different accelerator types ("cpu", "gpu", "tpu", "hpu", "mps", "auto")
                as well as custom accelerator instances.

            strategy: Supports different training strategies with aliases as well custom strategies.
                Default: ``"auto"``.

            devices: The devices to use. Can be set to a positive number (int or str), a sequence of device indices
                (list or str), the value ``-1`` to indicate all available devices should be used, or ``"auto"`` for
                automatic selection based on the chosen accelerator. Default: ``"auto"``.

            num_nodes: Number of GPU nodes for distributed training.
                Default: ``1``.

            precision: Double precision (64, '64' or '64-true'), full precision (32, '32' or '32-true'),
                16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed').
                Can be used on CPU, GPU, TPUs, or HPUs.
                Default: ``'32-true'``.

            logger: Logger (or iterable collection of loggers) for experiment tracking. A ``True`` value uses
                the default ``TensorBoardLogger`` if it is installed, otherwise ``CSVLogger``.
                ``False`` will disable logging. If multiple loggers are provided, local files
                (checkpoints, profiler traces, etc.) are saved in the ``log_dir`` of the first logger.
                Default: ``True``.

            callbacks: Add a callback or list of callbacks.
                Default: ``None``.

            fast_dev_run: Runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
                of train, val and test to find any bugs (ie: a sort of unit test).
                Default: ``False``.

            max_epochs: Stop training once this number of epochs is reached. Disabled by default (None).
                If both max_epochs and max_steps are not specified, defaults to ``max_epochs = 1000``.
                To enable infinite training, set ``max_epochs = -1``.

            min_epochs: Force training for at least these many epochs. Disabled by default (None).

            max_steps: Stop training after this number of steps. Disabled by default (-1). If ``max_steps = -1``
                and ``max_epochs = None``, will default to ``max_epochs = 1000``. To enable infinite training, set
                ``max_epochs`` to ``-1``.

            min_steps: Force training for at least these number of steps. Disabled by default (``None``).

            max_time: Stop training after this amount of time has passed. Disabled by default (``None``).
                The time duration can be specified in the format DD:HH:MM:SS (days, hours, minutes seconds), as a
                :class:`datetime.timedelta`, or a dictionary with keys that will be passed to
                :class:`datetime.timedelta`.

            limit_train_batches: How much of training dataset to check (float = fraction, int = num_batches).
                Default: ``1.0``.

            limit_val_batches: How much of validation dataset to check (float = fraction, int = num_batches).
                Default: ``1.0``.

            limit_test_batches: How much of test dataset to check (float = fraction, int = num_batches).
                Default: ``1.0``.

            limit_predict_batches: How much of prediction dataset to check (float = fraction, int = num_batches).
                Default: ``1.0``.

            overfit_batches: Overfit a fraction of training/validation data (float) or a set number of batches (int).
                Default: ``0.0``.

            val_check_interval: How often to check the validation set. Pass a ``float`` in the range [0.0, 1.0] to check
                after a fraction of the training epoch. Pass an ``int`` to check after a fixed number of training
                batches. An ``int`` value can only be higher than the number of training batches when
                ``check_val_every_n_epoch=None``, which validates after every ``N`` training batches
                across epochs or during iteration-based training.
                Default: ``1.0``.

            check_val_every_n_epoch: Perform a validation loop after every `N` training epochs. If ``None``,
                validation will be done solely based on the number of training batches, requiring ``val_check_interval``
                to be an integer value.
                Default: ``1``.

            num_sanity_val_steps: Sanity check runs n validation batches before starting the training routine.
                Set it to `-1` to run all batches in all validation dataloaders.
                Default: ``2``.

            log_every_n_steps: How often to log within steps.
                Default: ``50``.

            enable_checkpointing: If ``True``, enable checkpointing.
                It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
                :paramref:`~pytorch_lightning.trainer.trainer.Trainer.callbacks`.
                Default: ``True``.

            enable_progress_bar: Whether to enable to progress bar by default.
                Default: ``True``.

            enable_model_summary: Whether to enable model summarization by default.
                Default: ``True``.

            accumulate_grad_batches: Accumulates gradients over k batches before stepping the optimizer.
                Default: 1.

            gradient_clip_val: The value at which to clip gradients. Passing ``gradient_clip_val=None`` disables
                gradient clipping. If using Automatic Mixed Precision (AMP), the gradients will be unscaled before.
                Default: ``None``.

            gradient_clip_algorithm: The gradient clipping algorithm to use. Pass ``gradient_clip_algorithm="value"``
                to clip by value, and ``gradient_clip_algorithm="norm"`` to clip by norm. By default it will
                be set to ``"norm"``.

            deterministic: If ``True``, sets whether PyTorch operations must use deterministic algorithms.
                Set to ``"warn"`` to use deterministic algorithms whenever possible, throwing warnings on operations
                that don't support deterministic mode. If not set, defaults to ``False``. Default: ``None``.

            benchmark: The value (``True`` or ``False``) to set ``torch.backends.cudnn.benchmark`` to.
                The value for ``torch.backends.cudnn.benchmark`` set in the current session will be used
                (``False`` if not manually set). If :paramref:`~pytorch_lightning.trainer.trainer.Trainer.deterministic`
                is set to ``True``, this will default to ``False``. Override to manually set a different value.
                Default: ``None``.

            inference_mode: Whether to use :func:`torch.inference_mode` or :func:`torch.no_grad` during
                evaluation (``validate``/``test``/``predict``).

            use_distributed_sampler: Whether to wrap the DataLoader's sampler with
                :class:`torch.utils.data.DistributedSampler`. If not specified this is toggled automatically for
                strategies that require it. By default, it will add ``shuffle=True`` for the train sampler and
                ``shuffle=False`` for validation/test/predict samplers. If you want to disable this logic, you can pass
                ``False`` and add your own distributed sampler in the dataloader hooks. If ``True`` and a distributed
                sampler was already added, Lightning will not replace the existing one. For iterable-style datasets,
                we don't do this automatically.

            profiler: To profile individual steps during training and assist in identifying bottlenecks.
                Default: ``None``.

            detect_anomaly: Enable anomaly detection for the autograd engine.
                Default: ``False``.

            barebones: Whether to run in "barebones mode", where all features that may impact raw speed are
                disabled. This is meant for analyzing the Trainer overhead and is discouraged during regular training
                runs. The following features are deactivated:
                :paramref:`~pytorch_lightning.trainer.trainer.Trainer.enable_checkpointing`,
                :paramref:`~pytorch_lightning.trainer.trainer.Trainer.logger`,
                :paramref:`~pytorch_lightning.trainer.trainer.Trainer.enable_progress_bar`,
                :paramref:`~pytorch_lightning.trainer.trainer.Trainer.log_every_n_steps`,
                :paramref:`~pytorch_lightning.trainer.trainer.Trainer.enable_model_summary`,
                :paramref:`~pytorch_lightning.trainer.trainer.Trainer.num_sanity_val_steps`,
                :paramref:`~pytorch_lightning.trainer.trainer.Trainer.fast_dev_run`,
                :paramref:`~pytorch_lightning.trainer.trainer.Trainer.detect_anomaly`,
                :paramref:`~pytorch_lightning.trainer.trainer.Trainer.profiler`,
                :meth:`~pytorch_lightning.core.LightningModule.log`,
                :meth:`~pytorch_lightning.core.LightningModule.log_dict`.
            plugins: Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins.
                Default: ``None``.

            sync_batchnorm: Synchronize batch norm layers between process groups/whole world.
                Default: ``False``.

            reload_dataloaders_every_n_epochs: Set to a positive integer to reload dataloaders every n epochs.
                Default: ``0``.

            default_root_dir: Default path for logs and weights when no logger/ckpt_callback passed.
                Default: ``os.getcwd()``.
                Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'

            model_registry: The name of the model being uploaded to Model hub.

        Raises:
            TypeError:
                If ``gradient_clip_val`` is not an int or float.

            MisconfigurationException:
                If ``gradient_clip_algorithm`` is invalid.

        """
        super().__init__()
        self.trainer_kwargs_ = dict(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            logger=logger,
            callbacks=callbacks,
            fast_dev_run=fast_dev_run,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            max_steps=max_steps,
            min_steps=min_steps,
            max_time=max_time,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            limit_predict_batches=limit_predict_batches,
            overfit_batches=overfit_batches,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            num_sanity_val_steps=num_sanity_val_steps,
            log_every_n_steps=log_every_n_steps,
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            deterministic=deterministic,
            benchmark=benchmark,
            inference_mode=inference_mode,
            use_distributed_sampler=use_distributed_sampler,
            profiler=profiler,
            detect_anomaly=detect_anomaly,
            barebones=barebones,
            plugins=plugins,
            sync_batchnorm=sync_batchnorm,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            default_root_dir=default_root_dir
        )
        self.save_hyperparameters()

    def fit(self, train_dataloader, val_dataloader=None):
        """Fit the model on the training data."""
        _trainer = Trainer(**self.trainer_kwargs_)
        _trainer.fit(self, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        return self

    def predict(self, dataloader):
        check_is_fitted(self, "_trainer", check_is_none=True)
        outputs = self._trainer.predict(self, dataloaders=dataloader)
        return self.predict_epoch_end(outputs)

    def predict_step(self, *args, **kwargs):
        """Define the prediction step of the model."""
        return super().predict_step(*args, **kwargs)
    
    def predict_epoch_end(self, outputs):
        """Define post-processing on the outputs of predict_step"""
        return outputs
    
    def training_step(self, *args, **kwargs):
        return super().training_step(*args, **kwargs)
    
    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)
    
    def test_step(self, *args, **kwargs):
        return super().test_step(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        """Define the forward pass of the model."""
        return super().forward(*args, **kwargs)

    def log(self, *args, **kwargs):
        """Log a key, value metric."""
        return super().log(*args, **kwargs)
    
    def log_dict(self, *args, **kwargs):
        """Log a dictionary of metrics."""
        return super().log_dict(*args, **kwargs)


class EmbeddingTransformerMixin(ABC, LightningModule):
    """
    Mixing class for all embedding models in nidl. 
    
    It defines the following functionality:
    - 'transform' method for transforming input data into embeddings.
    - 'fit_transform' method for fitting the BaseEstimator to the data and transforming it.
    """


    def transform(self, loader):
        """ Get the embedding of the input data through the current model. 

        Parameters
        ----------
        loader: torch.utils.data.DataLoader
            Data loader generating batch of data that will be encoded by the model.
            Input data shape (n_samples, *)

        Returns
        ----------
        embedding: torch.Tensor
            Embedded tensor with shape (n_samples, n_embedding)
        """
        embedding = []
        self.freeze()

        for _, batch in enumerate(loader):
            emb = self.forward(batch)
            embedding.extend(emb)
        embedding = torch.stack(embedding, dim=0)

        self.unfreeze()
        return embedding
    
    def fit_transform(self, train_loader, val_loader=None):
        """ Fit the model to the training data and transform it into embeddings.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            Data loader generating batch of training data that will be used to fit the model.
            Input data shape (n_train_samples, *)

        val_loader: torch.utils.data.DataLoader, optional
            Data loader generating batch of validation data that will be used to validate the model.
            Input data shape (n_val_samples, *)

        Returns
        ----------
        embedding: torch.Tensor
            Embedded tensor with shape (n_train_samples, n_embedding)
        """
        self.fit(train_loader, val_loader=val_loader)
        return self.transform(train_loader)


