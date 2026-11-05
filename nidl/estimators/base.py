##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

import abc
import importlib
import inspect
import json
import tempfile
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.strategies import Strategy
from pytorch_lightning.trainer.connectors.accelerator_connector import (
    _PRECISION_INPUT,
)
from pytorch_lightning.utilities.types import (
    _METRIC,
    STEP_OUTPUT,
)
from torch.utils import data
from torchmetrics import MetricCollection

from ..info import __version__
from ..utils.validation import _estimator_is, available_if, check_is_fitted
from ..utils.weights import Weights

#: Modules in which the classes named by a configuration are looked up.
ENCODERS_MODULE = "nidl.backbones.volume"
PREPROCESSING_MODULE = "nidl.transforms.volume.preprocessing"

#: Parameters of the trainer saved with an estimator, with their default
#: values: they change how the estimator is trained (the "warmup_cosine"
#: schedule depends on `max_epochs`), unlike the hardware ones.
TRAINER_PARAMS = {
    "max_epochs": None,
    "max_steps": -1,
    "precision": None,
    "gradient_clip_val": None,
    "gradient_clip_algorithm": None,
    "accumulate_grad_batches": 1,
}


def _resolve(name: str, module_name: str, kind: str) -> Callable:
    """Return the public callable `name` of `module_name`.

    Configurations come from the Hub and are not trusted: only the objects
    exposed by nidl modules can be built, never an arbitrary import path.
    """
    module = importlib.import_module(module_name)
    obj = None
    if isinstance(name, str) and not name.startswith("_"):
        obj = getattr(module, name, None)
    if obj is None or inspect.ismodule(obj) or not callable(obj):
        raise ValueError(
            f"Unknown {kind} '{name}': it must be exposed by {module_name}."
        )
    return obj


def _preprocessing_config(preprocessing: Any) -> Any:
    """Check and copy a preprocessing entry, or a list of entries."""
    if not isinstance(preprocessing, Mapping):
        if not all(isinstance(entry, Mapping) for entry in preprocessing):
            raise ValueError(
                "The preprocessing must be a {'transform': ..., 'kwargs': "
                "...} entry or a list of them."
            )
        return [_preprocessing_config(entry) for entry in preprocessing]
    _resolve(preprocessing["transform"], PREPROCESSING_MODULE, "preprocessing")
    return {
        "transform": preprocessing["transform"],
        "kwargs": dict(preprocessing.get("kwargs", {})),
    }


def _fetch_file(
    model_id: Union[str, Path],
    filename: str,
    revision: Optional[str],
    token: Optional[Union[str, bool]],
    local_files_only: bool,
) -> Path:
    """Return the path of a file of a local folder or of a Hub repository."""
    path = Path(model_id)
    if path.is_dir():
        if not (path / filename).is_file():
            raise FileNotFoundError(f"No '{filename}' in '{path}'.")
        return path / filename
    try:
        from huggingface_hub.utils import GatedRepoError
    except ModuleNotFoundError:  # reported by `Weights.hf_download`
        GatedRepoError = ()
    try:
        return Weights.hf_download(
            None,
            str(model_id),
            filename,
            hf_revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
    except GatedRepoError as e:
        raise PermissionError(
            f"Access to the gated repository '{model_id}' has not been "
            f"granted to this account. Request it on "
            f"https://huggingface.co/{model_id}, then authenticate with "
            "`huggingface-cli login` or the `token` parameter."
        ) from e


def _load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"File '{path}' cannot be read because the 'safetensors' package "
            "is not installed. Please run 'pip install nidl[pretrained]' "
            "first."
        ) from e
    return load_file(str(path), device="cpu")


class NidlHubMixin:
    """Save, share and load nidl estimators with the Hugging Face Hub.

    An estimator is published as a folder, or a Hugging Face repository,
    holding two files:

    - `model.safetensors`: the weights of the estimator, named as in nidl;
    - `config.json`: everything needed to build the estimator before loading
      its weights.

    The configuration contains:

    - ``"estimator"``: the name of the estimator class;
    - ``"estimator_kwargs"``: the parameters of the estimator;
    - ``"trainer"`` (optional): the parameters of the trainer that change
      how the estimator is trained, such as `max_epochs` or `precision` (see
      ``TRAINER_PARAMS``), so that `fit` reproduces its training. The
      hardware ones, such as `devices`, are not saved;
    - ``"encoder"``: ``{"name": ..., "kwargs": ...}``, a backbone exposed by
      :mod:`nidl.backbones.volume` and its parameters, if the estimator has
      an encoder;
    - ``"input"`` (optional): ``{"shape": ..., "range": ...}``, the shape of
      a sample without batch dimension and the range of its values, checked
      by :meth:`check_pretrained_input` (either can be null);
    - ``"preprocessing"`` (optional): ``{"transform": ..., "kwargs": ...}``, a
      transform exposed by ``nidl.transforms.volume.preprocessing`` and
      the parameters fixed by the weights, or a list of them applied in
      order, see
      :meth:`~nidl.estimators.base.NidlHubMixin.get_preprocessing`;
    - ``"weights"`` (optional): ``{"missing_ok": [...]}``, prefixes of the
      parameters that the weights do not include and which keep their
      initial values, and for weights published elsewhere, their repository
      and converter (see :class:`~nidl.estimators.external.ExternalWeights`);
    - ``"nidl_version"``: the version of nidl that wrote it.

    Classes are looked up by name in these modules only, so that a
    configuration downloaded from the Hub cannot run arbitrary code.

    Weights that cannot be published again in the nidl format are loaded
    from the repository of their authors and renamed by a converter, as
    described by the ``"weights"`` entry of the configuration: see
    :class:`~nidl.estimators.external.ExternalWeights`.

    Attributes
    ----------
    pretrained_config_ : dict or None
        The configuration of the loaded weights, if any.
    """

    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "model.safetensors"

    pretrained_config_: Optional[dict[str, Any]] = None

    @classmethod
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        *,
        revision: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        local_files_only: bool = False,
        strict: bool = True,
        **kwargs: Any,
    ):
        """Build an estimator with published weights.

        Parameters
        ----------
        model_id : str or Path
            A Hugging Face repository, such as
            ``"nidl-neurospin/brainiac-vit-b"``, or a local folder, holding a
            `config.json` and a `model.safetensors`.
        revision : str or None, default=None
            Commit, branch or tag of the repository. By default, its last
            commit. Weights published elsewhere are always downloaded at the
            commit given by the configuration.
        token : str, bool or None, default=None
            Hugging Face token, needed for gated repositories. If None, the
            locally saved token or the `HF_TOKEN` variable is used.
        local_files_only : bool, default=False
            Whether to only use the local Hugging Face cache.
        strict : bool, default=True
            Whether the weights must match the estimator exactly. If False,
            a mismatch only raises a warning.
        **kwargs : dict
            Parameters of the estimator overriding the ones of the
            configuration, such as the `devices` or the `precision`.

        Returns
        -------
        estimator : object
            The estimator, ready for `transform`.

        Raises
        ------
        PermissionError
            If the repository is gated and access has not been granted.
        ValueError
            If the configuration is not for this estimator or cannot be
            read.
        RuntimeError
            If `strict` and the weights do not match the estimator.

        Examples
        --------
        >>> from nidl.estimators.ssl import SimCLR
        >>> model = SimCLR.from_pretrained("nidl-neurospin/brainiac-vit-b")
        >>> preprocess = model.get_preprocessing()
        """
        fetch_kwargs = {"token": token, "local_files_only": local_files_only}
        config_file = _fetch_file(
            model_id, cls.CONFIG_NAME, revision=revision, **fetch_kwargs
        )
        with open(config_file) as f:
            config = json.load(f)
        if "estimator" not in config:
            raise ValueError(f"'{model_id}' is not in the nidl format.")
        weights = config.get("weights", {})
        if "repo_id" in weights:  # published by their authors
            if not weights.get("revision"):
                raise ValueError(
                    f"The external weights of '{model_id}' need a revision."
                )
            weights_file = _fetch_file(
                weights["repo_id"],
                weights.get("filename", cls.WEIGHTS_NAME),
                revision=weights["revision"],
                **fetch_kwargs,
            )
        else:
            weights_file = _fetch_file(
                model_id, cls.WEIGHTS_NAME, revision=revision, **fetch_kwargs
            )
        state_dict = _load_safetensors(weights_file)
        if "converter" in weights:
            from . import external

            converter = _resolve(
                weights["converter"], external.__name__, "converter"
            )
            if not (
                isinstance(converter, type)
                and issubclass(converter, external.ExternalWeights)
                and converter is not external.ExternalWeights
            ):
                raise ValueError(
                    f"Unknown converter '{weights['converter']}'."
                )
            state_dict = converter().convert(state_dict)
        model = cls.from_config(config, **kwargs)
        model._load_pretrained_weights(state_dict, strict)
        model.fitted_ = True
        return model

    @classmethod
    def from_config(cls, config: Mapping[str, Any], **kwargs: Any):
        """Build an estimator from a configuration, without its weights.

        Parameters
        ----------
        config : mapping
            The configuration, see :class:`NidlHubMixin`.
        **kwargs : dict
            Parameters of the estimator or of its trainer overriding the
            ones of the configuration.

        Returns
        -------
        estimator : object
            A new, randomly initialized estimator.
        """
        if config.get("estimator") != cls.__name__:
            raise ValueError(
                f"This configuration is for {config.get('estimator')}, not "
                f"{cls.__name__}: use {config.get('estimator')} instead."
            )
        trainer = config.get("trainer", {})
        unknown = sorted(set(trainer) - set(TRAINER_PARAMS))
        if unknown:
            raise ValueError(f"Unknown trainer parameters {unknown}.")
        estimator_kwargs = {
            **trainer,
            **config.get("estimator_kwargs", {}),
            **kwargs,
        }
        encoder = config.get("encoder")
        if encoder is not None and "encoder" not in kwargs:
            build = _resolve(encoder["name"], ENCODERS_MODULE, "encoder")
            estimator_kwargs["encoder"] = build(**encoder.get("kwargs", {}))
        model = cls(**estimator_kwargs)
        model.pretrained_config_ = json.loads(json.dumps(config))
        return model

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        encoder: Optional[Mapping[str, Any]] = None,
        preprocessing: Optional[Mapping[str, Any]] = None,
        input_shape: Optional[Sequence[int]] = None,
        input_range: Optional[tuple[float, float]] = None,
    ) -> Path:
        """Save the estimator in the nidl format.

        The other parameters default to the ones of the loaded weights, if
        any, so that a fine-tuned estimator keeps them.

        Parameters
        ----------
        save_directory : str or Path
            The folder where `config.json` and `model.safetensors` are
            written, created if needed.
        encoder : mapping or None, default=None
            ``{"name": ..., "kwargs": ...}``, the backbone of
            :mod:`nidl.backbones.volume` building the encoder. It can be
            omitted if the estimator was given the encoder as a class and its
            `encoder_kwargs`.
        preprocessing : mapping, sequence of mapping or None, default=None
            ``{"transform": ..., "kwargs": ...}``, the transform of
            ``nidl.transforms.volume.preprocessing`` to apply on the
            inputs, with the parameters fixed by the weights, or a list of
            them applied in order.
        input_shape : sequence of int or None, default=None
            Shape of a sample, without batch dimension.
        input_range : (float, float) or None, default=None
            Range of the values of the inputs.

        Returns
        -------
        save_directory : Path
            The folder.
        """
        try:
            from safetensors.torch import save_file
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please run 'pip install nidl[pretrained]' to save weights."
            ) from e
        config = self._make_config(
            encoder, preprocessing, input_shape, input_range
        )
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                name: value.detach().cpu().clone().contiguous()
                for name, value in self.state_dict().items()
            },
            str(save_directory / self.WEIGHTS_NAME),
            metadata={"format": "pt"},
        )
        with open(save_directory / self.CONFIG_NAME, "w") as f:
            json.dump(config, f, indent=2)
        return save_directory

    def push_to_hub(
        self,
        repo_id: str,
        *,
        commit_message: str = "Upload nidl estimator",
        private: Optional[bool] = None,
        token: Optional[str] = None,
        **kwargs: Any,
    ):
        """Save the estimator in the nidl format to a Hub repository.

        Parameters
        ----------
        repo_id : str
            The repository, created if needed.
        commit_message : str, default="Upload nidl estimator"
            The message of the commit.
        private : bool or None, default=None
            Whether a new repository is private.
        token : str or None, default=None
            Hugging Face token. If None, the locally saved one is used.
        **kwargs : dict
            Parameters given to :meth:`save_pretrained`.

        Returns
        -------
        commit : huggingface_hub.CommitInfo
            The commit, whose `oid` can be given as `revision` to
            :meth:`from_pretrained`.
        """
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id, private=private, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            self.save_pretrained(tmp, **kwargs)
            return api.upload_folder(
                repo_id=repo_id,
                folder_path=tmp,
                commit_message=commit_message,
            )

    def get_preprocessing(self, **kwargs: Any):
        """Return the preprocessing the loaded weights expect.

        Parameters
        ----------
        **kwargs : dict
            Parameters of the transform depending on your data, such as the
            `input_orientation` of the volumes. They cannot override the
            parameters fixed by the weights. A preprocessing made of
            several transforms takes none.

        Returns
        -------
        transform : Transform, Compose or None
            The transform to apply on each sample, or None if the weights
            need none.

        Raises
        ------
        ValueError
            If a parameter is fixed by the weights.
        """
        entry = (self.pretrained_config_ or {}).get("preprocessing")
        if entry is None:
            return None
        if not isinstance(entry, Mapping):
            if kwargs:
                raise ValueError(
                    "This preprocessing is made of several transforms and "
                    f"takes no parameter, got {sorted(kwargs)}."
                )
            from torchvision.transforms import Compose

            return Compose([self._build_preprocessing(e, {}) for e in entry])
        return self._build_preprocessing(entry, kwargs)

    @staticmethod
    def _build_preprocessing(entry: Mapping[str, Any], kwargs: dict):
        transform = _resolve(
            entry["transform"], PREPROCESSING_MODULE, "preprocessing"
        )
        fixed = entry.get("kwargs", {})
        conflicts = sorted(set(fixed) & set(kwargs))
        if conflicts:
            raise ValueError(
                f"Parameters {conflicts} are fixed by the weights."
            )
        return transform(**fixed, **kwargs)

    def check_pretrained_input(self, x: torch.Tensor) -> None:
        """Check a batch against what the loaded weights were trained on.

        This is a no-op if the estimator does not come from
        :meth:`from_pretrained`.

        Parameters
        ----------
        x : torch.Tensor
            Batch of shape `(B, *input_shape)`.

        Raises
        ------
        ValueError
            If the batch has the wrong shape, contains non-finite values or
            is out of the expected range.
        """
        expected = (self.pretrained_config_ or {}).get("input")
        if expected is None:
            return
        help_msg = (
            "The loaded weights need a dedicated preprocessing, see "
            "`get_preprocessing`."
        )
        shape = expected.get("shape")
        if shape is not None and tuple(x.shape[1:]) != tuple(shape):
            raise ValueError(
                f"Expected samples of shape {tuple(shape)}, got "
                f"{tuple(x.shape[1:])}. {help_msg}"
            )
        if not torch.isfinite(x).all():
            raise ValueError(f"Input contains non-finite values. {help_msg}")
        value_range = expected.get("range")
        if value_range is not None:
            low, high = value_range
            x_min, x_max = x.min().item(), x.max().item()
            if x_min < low or x_max > high:
                raise ValueError(
                    f"Expected values in [{low}, {high}], got "
                    f"[{x_min}, {x_max}]. {help_msg}"
                )

    def _load_pretrained_weights(
        self, state_dict: Mapping[str, torch.Tensor], strict: bool
    ) -> None:
        weights = (self.pretrained_config_ or {}).get("weights", {})
        missing_ok = tuple(weights.get("missing_ok", ()))
        result = self.load_state_dict(state_dict, strict=False)
        missing = [
            key
            for key in result.missing_keys
            if not key.startswith(missing_ok)
        ]
        if missing or result.unexpected_keys:
            message = (
                f"The weights do not match {type(self).__name__}: missing "
                f"keys {missing}, unexpected keys {result.unexpected_keys}."
            )
            if strict:
                raise RuntimeError(message)
            warnings.warn(message, stacklevel=3)

    def _make_config(
        self,
        encoder: Optional[Mapping[str, Any]],
        preprocessing: Optional[Mapping[str, Any]],
        input_shape: Optional[Sequence[int]],
        input_range: Optional[tuple[float, float]],
    ) -> dict[str, Any]:
        previous = self.pretrained_config_ or {}
        config = {
            "nidl_version": __version__,
            "estimator": type(self).__name__,
            "estimator_kwargs": self._estimator_kwargs(),
        }
        trainer = {
            name: self.trainer_params_.get(name, default)
            for name, default in TRAINER_PARAMS.items()
            if self.trainer_params_.get(name, default) != default
        }
        if trainer:
            config["trainer"] = trainer
        encoder = encoder or previous.get("encoder") or self._encoder_config()
        if encoder is not None:
            _resolve(encoder["name"], ENCODERS_MODULE, "encoder")
            config["encoder"] = {
                "name": encoder["name"],
                "kwargs": dict(encoder.get("kwargs", {})),
            }
        if input_shape is not None or input_range is not None:
            config["input"] = {
                "shape": None if input_shape is None else list(input_shape),
                "range": None if input_range is None else list(input_range),
            }
        elif "input" in previous:
            config["input"] = previous["input"]
        preprocessing = preprocessing or previous.get("preprocessing")
        if preprocessing is not None:
            config["preprocessing"] = _preprocessing_config(preprocessing)
        try:
            return json.loads(json.dumps(config))
        except TypeError as e:
            raise TypeError(
                f"The configuration cannot be saved in JSON: {e}."
            ) from e

    def _estimator_kwargs(self) -> dict[str, Any]:
        """Return the parameters of the estimator, without the trainer's."""
        excluded = set(inspect.signature(BaseEstimator.__init__).parameters)
        excluded |= {"self", "encoder", "encoder_kwargs"}
        signature = inspect.signature(type(self).__init__).parameters
        kwargs = {}
        for name, value in self.hparams.items():
            if name not in signature or name in excluded:
                continue
            try:
                json.dumps(value)
            except TypeError as e:
                if value is signature[name].default:
                    continue
                raise TypeError(
                    f"Parameter '{name}' of {type(self).__name__} cannot be "
                    f"saved in JSON: {value!r}."
                ) from e
            kwargs[name] = value
        return kwargs

    def _encoder_config(self) -> Optional[dict[str, Any]]:
        """Return the encoder of the hyper-parameters, if given as a class."""
        signature = inspect.signature(type(self).__init__).parameters
        if "encoder" not in signature:
            return None
        encoder = self.hparams.get("encoder")
        if isinstance(encoder, type):
            try:
                if _resolve(encoder.__name__, ENCODERS_MODULE, "") is encoder:
                    kwargs = self.hparams.get("encoder_kwargs") or {}
                    return {"name": encoder.__name__, "kwargs": dict(kwargs)}
            except ValueError:
                pass
        raise ValueError(
            "The encoder cannot be saved: give `encoder`, such as "
            "{'name': 'resnet18', 'kwargs': {...}}, with a backbone exposed "
            f"by {ENCODERS_MODULE}."
        )


class BaseEstimator(NidlHubMixin, pl.LightningModule):
    """Base class for all estimators in the NIDL framework designed for
    scalability.

    Inherits from PyTorch Lightning's LightningModule.
    This class provides a common interface for training, validation, and
    prediction/transformation in a distributed setting (multi-node multi-GPU)
    inheriting from the Lightning's Trainer capabilities.

    It also provides a common interface for logging metrics, saving
    hyperparameters and handling callbacks, which are modular pieces of code
    that hook into the training loop without modifying your model code. They
    can be used for a variety of purposes, such as early stopping, model
    checkpointing, model's probing, metrics computations, etc.

    This class provides:

    - a :meth:`fit` method.
    - :meth:`~nidl.estimators.base.NidlHubMixin.from_pretrained`,
      :meth:`~nidl.estimators.base.NidlHubMixin.save_pretrained` and
      :meth:`~nidl.estimators.base.NidlHubMixin.push_to_hub` methods, to
      share estimators and their weights with the Hugging Face Hub (see
      :class:`~nidl.estimators.base.NidlHubMixin`).
    - a :meth:`transform` if the child class inherits from
      :class:`TransformerMixin`
    - a :meth:`transform_with_targets` method if the child class inherits from
      :class:`TransformerMixin`
    - a :meth:`predict` method if the child class inherits from
      :class:`RegressorMixin` (for regression) or :class:`ClassifierMixin`
      (for classification) or :class:`ClusterMixin` (for clustering)

    Parameters
    ----------
    callbacks: list of Callback or Callback, default=None
        add a callback or list of callbacks.
    check_val_every_n_epoch: int, default=1
        perform a validation loop after every `N` training epochs. If ``None``,
        validation will be done solely based on the number of training
        batches, requiring ``val_check_interval`` to be an integer value.
    val_check_interval: int or float, default=None
        how often to check the validation set. Pass a ``float`` in the range
        [0.0, 1.0] to check after a fraction of the training epoch. Pass an
        ``int`` to check after a fixed number of training batches. An ``int``
        value can only be higher than the number of training batches when
        ``check_val_every_n_epoch=None``, which validates after every ``N``
        training batches across epochs or during iteration-based training.
        Default: ``1.0``.
    num_sanity_val_steps: int, default=None
        Sanity check runs n validation batches before starting the training
        routine. Set it to `-1` to run all batches in all validation
        dataloaders. Default: ``2``.
    max_epochs: int, default=None
        stop training once this number of epochs is reached. If both
        max_epochs and max_steps are not specified, defaults to
        ``max_epochs = 1000``. To enable infinite training, set
        ``max_epochs = -1``.
    min_epochs: int, default=None
        force training for at least these many epochs.
        Disabled by default.
    max_steps: int, default -1
        stop training after this number of steps. If ``max_steps = -1``
        and ``max_epochs = None``, will default to ``max_epochs = 1000``.
        To enable infinite training, set ``max_epochs`` to ``-1``.
    min_steps: int, default=None
        force training for at least these number of steps.
        Disabled by default.
    enable_checkpointing: bool, default=None
        if ``True``, enable checkpointing. It will configure a default
        ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
        trainer callbacks.
        Default: ``True``.
    enable_progress_bar: bool, default=None
        whether to enable to progress bar by default.
        Default: ``True``.
    enable_model_summary: bool, default=None
        whether to enable model summarization by default.
        Default: ``True``.
    accelerator: str or Accelerator, default="auto"
        supports passing different accelerator types ("cpu", "gpu", "tpu",
        "hpu", "mps", "auto") as well as custom accelerator instances.
    strategy: str or Strategy, default="auto"
        supports different training strategies with aliases as well custom
        strategies.
    devices: listof int, str, int, default="auto"
        the devices to use. Can be set to a positive number (int or str), a
        sequence of device indices (list or str), the value ``-1`` to indicate
        all available devices should be used, or ``"auto"`` for automatic
        selection based on the chosen accelerator.
    num_nodes: int, default=1
        number of GPU nodes for distributed training.
    precision: int or str, default="32-true"
        double precision (64, '64' or '64-true'), full
        precision (32, '32' or '32-true'), 16bit mixed precision (16, '16',
        '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed').
        Can be used on CPU, GPU, TPUs, or HPUs.
    save_hparams: bool, default=True
        Whether to save the hyper-parameters of this estimator or not.
    ignore: list of str, default=None
        Attributes to be ignored when saving the hyperparameters of the
        estimator. This is particularly useful for ignoring
        :class:`~torch.nn.Module` attributes and callbacks.
    random_state: int, default=None
        when shuffling is used, `random_state` affects the ordering of the
        indices, which controls the randomness of each batch. Pass an
        int for reproducible output across multiple function calls.
    kwargs: dict
        lightning's trainer extra parameters.

    Attributes
    ----------
    fitted
        a boolean that is True if the estimator has been fitted, and is False
        otherwise.
    hparams:
        contains the estimator hyperparameters.
    trainer
        the current lightning trainer.
    trainer_params
        a dictionaray with the trainer parameters.

    Notes
    -----
    Callbacks can help you to tune, monitor or debug an estimator. For
    instance you can check the type of the input batches using
    `BatchTypingCallback` callback.
    """

    def __init__(
        self,
        callbacks: Optional[Union[list[Callback], Callback]] = None,
        check_val_every_n_epoch: Optional[int] = 1,
        val_check_interval: Optional[Union[int, float]] = None,
        num_sanity_val_steps: Optional[int] = None,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        max_steps: int = -1,
        min_steps: Optional[int] = None,
        enable_checkpointing: Optional[bool] = None,
        enable_progress_bar: Optional[bool] = None,
        enable_model_summary: Optional[bool] = None,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[list[int], str, int] = "auto",
        num_nodes: int = 1,
        precision: Optional[_PRECISION_INPUT] = None,
        save_hparams: bool = True,
        ignore: Optional[Sequence[str]] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        if save_hparams:
            self.save_hyperparameters(ignore=ignore)
        self.fitted_ = False
        self.trainer_params_ = {
            "callbacks": callbacks,
            "check_val_every_n_epoch": check_val_every_n_epoch,
            "num_sanity_val_steps": num_sanity_val_steps,
            "val_check_interval": val_check_interval,
            "max_epochs": max_epochs,
            "min_epochs": min_epochs,
            "max_steps": max_steps,
            "min_steps": min_steps,
            "enable_checkpointing": enable_checkpointing,
            "enable_progress_bar": enable_progress_bar,
            "enable_model_summary": enable_model_summary,
            "accelerator": accelerator,
            "strategy": strategy,
            "devices": devices,
            "num_nodes": num_nodes,
            "precision": precision,
        }
        self.trainer_params_.update(kwargs)

    @property
    def fitted(self):
        return self.fitted_

    @property
    def trainer_params(self):
        return self.trainer_params_

    def fit(
        self,
        train_dataloader: Optional[data.DataLoader] = None,
        val_dataloader: Optional[data.DataLoader] = None,
        datamodule: Optional[pl.LightningDataModule] = None,
        ckpt_path: Union[str, Path, None] = None,
        weights_only: Optional[bool] = None,
    ):
        """The `fit` method.

        In the child class you will need to define:

        - a `training_step` method for defining the training instructions at
          each step.
        - a `validation_step` method for defining the validation instructions
          at each step.

        Parameters
        ----------
        train_dataloader: torch DataLoader, default=None
            The training data loader. If ``datamodule`` is passed, the
            ``train_dataloader`` hook is used instead, ignoring this loader.
        val_dataloader: torch DataLoader, default=None
            The validation data loader. If ``datamodule`` is passed, the
            ``val_dataloader`` hook is used instead, ignoring this loader.
        datamodule: pl.LightningDataModule, default=None
            An instance of `LightningDataModule`, alternative to passing
            `train_dataloader`/`val_dataloader`.
        ckpt_path: str, Path, default=None
            Path to a checkpoint from which training is resumed.
        weights_only: bool, default=None
            If True, restricts `torch.load` (used to load the checkpoint)
            to loading only tensors, primitive types and dictionaries,
            without executing arbitrary code. Passed to
            `torch.load(weights_only=...)`.

        Returns
        -------
        self: object
            fitted estimator.
        """
        trainer = pl.Trainer(**self.trainer_params_)
        if trainer.logger is not None:
            trainer.logger._default_hp_metric = None
        pl.seed_everything(self.hparams.random_state)
        trainer.fit(
            self,
            train_dataloader,
            val_dataloader,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            weights_only=weights_only,
        )
        self.fitted_ = True
        return self

    @available_if(_estimator_is("transformer"))
    def transform(self, test_dataloader: data.DataLoader) -> Any:
        """The `transform` method for transformer.

        In the child class you will need to define:

        - a `transform_step` method for defining the transform instructions at
          each step.

        Parameters
        ----------
        test_dataloader: torch DataLoader
            testing samples.

        Returns
        -------
        out: torch Tensor
            returns transformed samples.

        Notes
        -----
        Since we expect a tensor as output of the model, it is gathered across
        GPUs in the multi-gpu distributed case and the output is stored on the
        relevant device defined by the trainer's strategy (cpu or gpu).
        """
        check_is_fitted(self)
        trainer = pl.Trainer(**self.trainer_params_)
        f = torch.cat(
            trainer.predict(self, test_dataloader, return_predictions=True)
        )
        # Predictions are moved to cpu by default in PL
        # => mv it again on GPU if necessary
        f = f.to(trainer.strategy.root_device)
        # Gather transformations across GPUs
        f = trainer.strategy.all_gather(f)
        # Reduce (world_size, batch, ...) to (world_size * batch, ...)
        if f.ndim > 2 and trainer.world_size > 1:
            f = f.reshape(f.shape[0] * f.shape[1], *f.shape[2:])
        return f

    @available_if(_estimator_is("transformer"))
    def transform_with_targets(
        self,
        test_dataloader: data.DataLoader,
    ) -> tuple[Any, Any]:
        """Return transformed samples and associated targets.

        A default :meth:`transform_step_with_targets` method is available
        but it can be overrides to define your own logic.

        Parameters
        ----------
        test_dataloader: torch DataLoader
            Testing samples. By default, the form expected is ``(x, y)``
            where ``y`` is the target and ``x`` the input. This logic can
            be overrides by rewriting your own
            :meth:`transform_step_with_targets`.

        Returns
        -------
        features, targets: torch Tensor, torch.Tensor
            Transformed samples and associated targets.

        Notes
        -----
        Outputs are gathered across GPUs in distributed settings, similarly to
        :meth:`~nidl.estimators.BaseEstimator.transform`.
        """
        check_is_fitted(self)
        trainer = pl.Trainer(**self.trainer_params_)

        self._return_targets_in_predict = True
        try:
            outputs = trainer.predict(
                self, test_dataloader, return_predictions=True
            )
        finally:
            self._return_targets_in_predict = False

        features = torch.cat([out["features"] for out in outputs])
        targets = torch.cat([out["targets"] for out in outputs])

        features = features.to(trainer.strategy.root_device)
        targets = targets.to(trainer.strategy.root_device)

        features = trainer.strategy.all_gather(features)
        targets = trainer.strategy.all_gather(targets)

        if features.ndim > 2 and trainer.world_size > 1:
            features = features.reshape(
                features.shape[0] * features.shape[1], *features.shape[2:]
            )

        if targets.ndim > 1 and trainer.world_size > 1:
            targets = targets.reshape(
                targets.shape[0] * targets.shape[1], *targets.shape[2:]
            )

        return features, targets

    @available_if(_estimator_is(("regressor", "classifier", "clusterer")))
    def predict(self, test_dataloader: data.DataLoader) -> Any:
        """The `predict` method for regression, classification and clustering.

        In the child class you will need to define:

        - a `predict_step` method for defining the predict instructions at
          each step.

        Parameters
        ----------
        test_dataloader: torch DataLoader
            testing samples.

        Returns
        -------
        out: torch Tensor
            returns predicted samples.
        """
        check_is_fitted(self)
        trainer = pl.Trainer(**self.trainer_params_)
        return torch.cat(
            trainer.predict(self, test_dataloader, return_predictions=True)
        )

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = 0
    ) -> STEP_OUTPUT:
        """Here you compute and return the training loss and some additional
        metrics for e.g. the progress bar or logger.

        Parameters
        ----------
        batch: Any
            Output of your training loader iterable, normally a
            :class:`~torch.utils.data.DataLoader`
        batch_idx: int
            the index of this batch.
        dataloader_idx: int, default=0
            the index of the dataloader that produced this batch (only if
            multiple dataloaders are used).

        Returns
        -------
        loss: STEP_OUTPUT
            the computed loss:

            - :class:`~torch.Tensor` - the loss tensor.
            - ``dict`` - a dictionary which can include any keys, but must
              include the key ``'loss'`` in the case of automatic optimization.
            - ``None`` - in automatic optimization, this will skip to the
              next batch (but is not supported for multi-GPU, TPU, or
              DeepSpeed). For manual optimization, this has no special
              meaning, as returning the loss is not required.

        To use multiple optimizers, you can switch to 'manual optimization'
        and control their stepping:

        Examples
        --------
        >>> def __init__(self):
        >>>     super().__init__()
        >>>     self.automatic_optimization = False
        >>>
        >>>
        >>> # Multiple optimizers (e.g.: GANs)
        >>> def training_step(self, batch, batch_idx):
        >>>     opt1, opt2 = self.optimizers()
        >>>
        >>>     # do training_step with encoder
        >>>     ...
        >>>     opt1.step()
        >>>     # do training_step with decoder
        >>>     ...
        >>>     opt2.step()

        Notes
        -----
        When ``accumulate_grad_batches`` > 1, the loss returned here will be
        automatically normalized by ``accumulate_grad_batches`` internally.
        """
        return super().training_step(batch, batch_idx, dataloader_idx)

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = 0
    ) -> STEP_OUTPUT:
        """Operates on a single batch of data from the validation set. In
        this step you'd might generate examples or calculate anything of
        interest like accuracy.

        Parameters
        ----------
        batch: Any
            Output of your validation loader iterable, normally a
            :class:`~torch.utils.data.DataLoader`
        batch_idx: int
            the index of this batch.
        dataloader_idx: int, default=0
            the index of the dataloader that produced this batch (only if
            multiple dataloaders are used).

        Returns
        -------
        loss: STEP_OUTPUT
            the computed loss:

            - :class:`~torch.Tensor` - the loss tensor.
            - ``dict`` - a dictionary. can include any keys, but must include
              the key ``'loss'``.
            - ``None`` - skip to the next batch.

        Notes
        -----
        If you don't need to validate you don't need to implement this method.


        When the :meth:`validation_step` is called, the model has been put in
        eval mode and PyTorch gradients have been disabled. At the end of
        validation,  the model goes back to training mode and gradients are
        enabled.
        """
        return super().validation_step(batch, batch_idx, dataloader_idx)

    @available_if(_estimator_is("transformer"))
    @abc.abstractmethod
    def transform_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = 0
    ) -> Any:
        """Define a transform step.

        Share the same API as :meth:`BaseEstimator.predict_step`.
        """

    @available_if(_estimator_is("transformer"))
    def transform_step_with_targets(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ) -> Mapping[str, Any]:
        """Step function called during :meth:`~transform_with_targets`.

        :meth:`~transform_with_targets` is used to scale the inference step of
        embedding estimators to multi-devices.

        Parameters
        ----------
        batch: Any
            Output of your test loader iterable, normally a
            :class:`~torch.utils.data.DataLoader`. It expects batches of
            the form ``(x, y)`` where only ``x`` is processed by
            the :meth:`~transform_step` and ``y`` is returned as-is.
        batch_idx: int
            the index of this batch.
        dataloader_idx: int, default=0
            the index of the dataloader that produced this batch (only if
            multiple dataloaders are used).

        Returns
        -------
        dict[str, Any]
            Dictionary with keys ``"features"`` and ``"targets"`` and values
            should be ``torch.Tensor``.
        """
        if not isinstance(batch, Sequence) or len(batch) < 2:
            raise ValueError(
                "`transform_with_targets` expects batches with at least two "
                "elements, typically `(x, y)`."
            )
        x, y = batch[:2]

        features = self.transform_step(x, batch_idx, dataloader_idx)
        return {"features": features, "targets": y}

    @available_if(
        _estimator_is(("transformer", "regressor", "classifier", "clusterer"))
    )
    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = 0
    ) -> Any:
        """Step function called during
        :meth:`~lightning.pytorch.trainer.trainer.Trainer.predict`. By default,
        it calls :meth:`~lightning.pytorch.core.LightningModule.forward`.
        Override to add any processing logic.

        The :meth:`~lightning.pytorch.core.LightningModule.predict_step` is
        used to scale inference on multi-devices.

        To prevent an OOM error, it is possible to use
        :class:`~lightning.pytorch.callbacks.BasePredictionWriter`
        callback to write the predictions to disk or database after each
        batch or on epoch end.

        The :class:`~lightning.pytorch.callbacks.BasePredictionWriter` should
        be used while using a spawn based accelerator. This happens for
        training strategy ``strategy="ddp_spawn"`` or training on 8 TPU cores
        with ``accelerator="tpu", devices=8`` as predictions won't be returned.

        Parameters
        ----------
        batch: iterable, normally a :class:`~torch.utils.data.DataLoader`
            the current data.
        batch_idx: int
            the index of this batch.
        dataloader_idx: int, default=0
            the index of the dataloader that produced this batch (only if
            multiple dataloaders are used).

        Returns
        -------
        out: Any
            the predicted output.
        """
        if _estimator_is("transformer"):
            if getattr(self, "_return_targets_in_predict", False):
                return self.transform_step_with_targets(
                    batch, batch_idx, dataloader_idx
                )
            return self.transform_step(batch, batch_idx, dataloader_idx)
        else:
            return super().predict_step(batch, batch_idx, dataloader_idx)

    def log(
        self,
        name: str,
        value: _METRIC,
        prog_bar: bool = False,
        logger: Optional[bool] = None,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        reduce_fx: Union[str, Callable] = "mean",
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_group: Optional[Any] = None,
        add_dataloader_idx: bool = True,
        batch_size: Optional[int] = None,
        metric_attribute: Optional[str] = None,
        rank_zero_only: bool = False,
    ) -> None:
        """Log a key, value pair.

        Parameters
        ----------
        name: str
            key to log. Must be identical across all processes if using DDP or
            any other distributed strategy.
        value: object
            value to log. Can be a ``float``, ``Tensor``, or a ``Metric``.
        prog_bar: bool, default=False
            if ``True`` logs to the progress bar.
        logger: bool, default=None
            if ``True`` logs to the logger.
        on_step: bool, default=None
            if ``True`` logs at this step. The default value is determined by
            the hook.
        on_epoch: bool, default=None
            if ``True`` logs epoch accumulated metrics. The default value is
            determined by the hook.
        reduce_fx: str of callable, default='mean'
            reduction function over step values for end of epoch.
            :func:`torch.mean` by default.
        enable_graph: bool, default=False
            if ``True``, will not auto detach the graph.
        sync_dist: bool, default=False
            if ``True``, reduces the metric across devices. Use with care as
            this may lead to a significant communication overhead.
        sync_dist_group: object, default=None
            the DDP group to sync across.
        add_dataloader_idx: bool, default=True
            if ``True``, appends the index of the current dataloader to
            the name (when using multiple dataloaders). If ``False``, user
            needs to give unique names for each dataloader to not mix the
            values.
        batch_size: int, default=None
            current batch_size. This will be directly inferred from the
            loaded batch, but for some data structures you might need to
            explicitly provide it.
        metric_attribute: str, default=None
            to restore the metric state, Lightning requires the reference of
            the :class:`torchmetrics.Metric` in your model. This is found
            automatically if it is a model attribute.
        rank_zero_only: bool, default=False
            tells Lightning if you are calling ``self.log`` from every
            process (default) or only from rank 0. If ``True``, you won't be
            able to use this metric as a monitor in callbacks (e.g., early
            stopping).
            Warning: Improper use can lead to deadlocks!

        Examples
        --------
        >>> self.log('train_loss', loss)
        """
        return super().log(
            name=name,
            value=value,
            prog_bar=prog_bar,
            logger=logger,
            on_step=on_step,
            on_epoch=on_epoch,
            reduce_fx=reduce_fx,
            enable_graph=enable_graph,
            sync_dist=sync_dist,
            sync_dist_group=sync_dist_group,
            add_dataloader_idx=add_dataloader_idx,
            batch_size=batch_size,
            metric_attribute=metric_attribute,
            rank_zero_only=rank_zero_only,
        )

    def log_dict(
        self,
        dictionary: Union[Mapping[str, _METRIC], MetricCollection],
        prog_bar: bool = False,
        logger: Optional[bool] = None,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        reduce_fx: Union[str, Callable] = "mean",
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_group: Optional[Any] = None,
        add_dataloader_idx: bool = True,
        batch_size: Optional[int] = None,
        rank_zero_only: bool = False,
    ) -> None:
        """Log a dictionary of values at once.

        Parameters
        ----------
        dictionary: dict
            key value pairs. Keys must be identical across all processes if
            using DDP or any other distributed strategy.
            The values can be a ``float``, ``Tensor``, ``Metric``, or
            ``MetricCollection``.
        prog_bar: bool, default=False
            if ``True`` logs to the progress bar.
        logger: bool, default=None
            if ``True`` logs to the logger.
        on_step: bool, default=None
            ``None`` auto-logs for training_step but not validation/test_step.
            The default value is determined by the hook.
        on_epoch: bool, default=None
            ``None`` auto-logs for val/test step but not ``training_step``.
            The default value is determined by the hook.
        reduce_fx: str of callable, default='mean'
            reduction function over step values for end of epoch.
            :func:`torch.mean` by default.
        enable_graph: bool, default=False
            if ``True``, will not auto detach the graph.
        sync_dist: bool, default=False
            if ``True``, reduces the metric across devices. Use with care as
            this may lead to a significant communication overhead.
        sync_dist_group: object, default=None
            the DDP group to sync across.
        add_dataloader_idx: bool, default=True
            if ``True``, appends the index of the current dataloader to
            the name (when using multiple dataloaders). If ``False``, user
            needs to give unique names for each dataloader to not mix the
            values.
        batch_size: int, default=None
            current batch_size. This will be directly inferred from the
            loaded batch, but for some data structures you might need to
            explicitly provide it.
        rank_zero_only: bool, default=False
            tells Lightning if you are calling ``self.log`` from every
            process (default) or only from rank 0. If ``True``, you won't be
            able to use this metric as a monitor in callbacks (e.g., early
            stopping).
            Warning: Improper use can lead to deadlocks!

        Examples
        --------
        >>> values = {'loss': loss, 'acc': acc, ..., 'metric_n': metric_n}
        >>> self.log_dict(values)
        """
        return super().log_dict(
            dictionary=dictionary,
            prog_bar=prog_bar,
            logger=logger,
            on_step=on_step,
            on_epoch=on_epoch,
            reduce_fx=reduce_fx,
            enable_graph=enable_graph,
            sync_dist=sync_dist,
            sync_dist_group=sync_dist_group,
            add_dataloader_idx=add_dataloader_idx,
            batch_size=batch_size,
            rank_zero_only=rank_zero_only,
        )

    def on_load_checkpoint(self, checkpoint):
        """Hook that is called when using the `load_from_checkpoint` method."""
        self.fitted_ = True
        return super().on_load_checkpoint(checkpoint)


class RegressorMixin:
    """Mixin class for all regression estimators in nidl.

    This mixin sets the estimator type to `"regressor"` through the
    `estimator_type` tag.
    """

    _estimator_type = "regressor"


class ClassifierMixin:
    """Mixin class for all classifiers in nidl.

    This mixin sets the estimator type to `classifier` through the
    `estimator_type` tag.
    """

    _estimator_type = "classifier"


class ClusterMixin:
    """Mixin class for all cluster estimators in nidl.

    This mixin sets the estimator type to `clusterer` through the
    `estimator_type` tag.
    """

    _estimator_type = "clusterer"


class TransformerMixin:
    """Mixin class for all transformers in nidl.

    This mixin sets the estimator type to `transformer` through the
    `estimator_type` tag.
    """

    _estimator_type = "transformer"
