##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Hugging face utilities.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytorch_lightning as pl
import torch
from huggingface_hub import (
    HfApi,
    file_exists,
    hf_hub_download,
)
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from safetensors.torch import (
    load_file,
    save_file,
)

from .config import (
    ConfigLoader,
)
from .validation import (
    check_is_fitted,
    get_cache_dir,
)


class HubMixin:
    """
    Save, share, and load nidl estimators with the Hugging Face Hub.

    An estimator is published as a local directory or a Hugging Face repository
    holding two files:

    - ``model.safetensors``: the weights of the fitted estimator;
    - ``config.yaml``: the Hydra instantiation configuration capturing all
      constructor arguments needed to rebuild the training data and estimator
      before loading its weights.

    To enable automatic serialization via :meth:`save_pretrained` or
    :meth:`push_pretrained`, estimators must be instantiated inside a
    :meth:`ConfigLoader.record` context manager (or decorated with
    :meth:`ConfigLoader.record_config`) so their outermost ``__init__``
    arguments are recorded in ``_init_args``.

    When reloading via :meth:`from_pretrained`, the configuration file is
    fetched from the Hub and instantiated with :meth:`ConfigLoader.instantiate`
    into a :class:`types.SimpleNamespace` containing ``estimator`` (and any
    other configured components). If a ``weights`` component is present, the
    specified weights are loaded into the ``estimator``. Otherwise,
    the ``model.safetensors`` next to the configuration file are used.

    Raises
    ------
    ValueError
        If no weights are found.

    See Also
    --------
    ConfigLoader.record : Context manager that records constructor arguments
        for subclass hierarchies during instantiation.
    ConfigLoader.instantiate : Instantiate a saved YAML configuration file.

    Examples
    --------
    >>> from pathlib import Path
    >>> import torch
    >>> import torch.nn as nn
    >>> from torch.utils.data import DataLoader
    >>> from torchvision.transforms import v2
    >>> from nidl.datasets import OpenBHB
    >>> from nidl.estimators.ssl import SimCLR
    >>> from nidl.utils.config import ConfigLoader
    >>> from nidl.utils.hub import Weights

    Capture an estimator at instantiation, fit it and save it locally:

    >>> with ConfigLoader.record(nn.Module, Weights):
    ...     estimator = SimCLR(encoder=nn.Linear(128, 64), lr=1e-3)
    ...     estimator.fitted_ = True
    ...     cfg_file, weights_file = estimator.save_pretrained(
    ...         repo_id="local/v1",
    ...     )
    >>> print(cfg_file.read_text(encoding="utf-8"))

    Create a new release with the data laoder included:

    >>> with ConfigLoader.record(nn.Module, OpenBHB, DataLoader, Weights):
    ...     dataloader = DataLoader(
    ...         OpenBHB(
    ...             Path(__file__).parent / "nidl_data",
    ...             modality="vbm_roi",
    ...             target=None,
    ...             transforms=v2.Compose([
    ...                 v2.ToDtype(torch.float32, scale=True),
    ...             ]),
    ...             streaming=True,
    ...         ),
    ...         batch_size=3,
    ...         num_workers=1,
    ...         shuffle=True,
    ...     )
    ...     cfg_file, weights_file = estimator.save_pretrained(
    ...         repo_id="local/v2",
    ...         recorded_instances={"train_dataloader": dataloader}
    ...     )
    >>> print(cfg_file.read_text(encoding="utf-8"))

    Reload a pretrained estimator and train data laoder:

    >>> loaded = SimCLR.from_pretrained(
    ...     "local/v2",
    ...     local_files_only=True,
    ... )
    """
    CONFIG_NAME = "config.yaml"
    WEIGHTS_NAME = "model.safetensors"

    @classmethod
    def from_pretrained(
            cls,
            repo_id: str,
            *,
            revision: str | None = None,
            token: str | None = None,
            local_files_only: bool = False,
            strict: bool = True,
            data_dir: str | Path | None = None,
            **kwargs: Any,
        ) -> SimpleNamespace:
        """
        Build an estimator from a configuration file stored on a Hugging Face
        repository.

        Parameters
        ----------
        repo_id : str
            A Hugging Face repository, such as
            ``"username/repo-name"``.
        revision : str | None
            Commit tag of the repository. If None the last commit is used.
            Default None.
        token : str | None
            Hugging Face token, needed for gated repositories. If None, the
            locally saved token or the `HF_TOKEN` variable is used.
            Default None.
        local_files_only : bool
            Whether to only use the local Hugging Face cache.
            Default False.
        strict : bool
            Whether the weights must match the estimator exactly. If False,
            a mismatch only raises a warning.
            Default True.
        data_dir : str | Path | None
            Path where data should be downloaded. By default, files are
            downloaded in the standard nidl cache directory.
            Default None.
        **kwargs : dict
            Parameters of the estimator overriding the ones of the
            configuration, such as the `devices` or the `precision`.

        Returns
        -------
        estimator : BaseEstimator
            A configured  estimator.

        Raises
        ------
        PermissionError
            If the repository is gated and access has not been granted.
        ValueError
            If the configuration is not for this estimator or cannot be
            read.
        RuntimeError
            If `strict` and the weights do not match the estimator.

        Notes
        -----
        Force any parameters except ``repo_id`` to be passed by name (keyword)
        rather than by position.
        """
        config_file = fetch_file(
            repo_id,
            cls.CONFIG_NAME,
            revision=revision,
            token=token,
            data_dir=data_dir,
            local_files_only=local_files_only,
        )
        cfg_instantiated = ConfigLoader.instantiate(
            config_file,
            data_dir=data_dir,
        )
        estimator = getattr(cfg_instantiated, "estimator", None)
        weights = getattr(cfg_instantiated, "weights", None)
        if weights is None:
            weights = Weights(
                filepath=cls.WEIGHTS_NAME,
                repo_id=repo_id,
                revision=revision,
                token=token,
                data_dir=data_dir,
                local_files_only=local_files_only,
            )
        else:
            del cfg_instantiated.weights
        if estimator is not None:
            weights.load_pretrained(
                estimator,
                weights_only=True,
                strict=strict,
            )
            estimator.fitted_ = True
        return cfg_instantiated

    def save_pretrained(
            self,
            repo_id: str,
            *,
            data_dir: str | Path | None = None,
            recorded_instances: dict[str, Any] | None = None,
        ) -> Path:
        """
        Save an estimator.

        The weights are added in the saved configuration as well as the given
        external instances.

        Parameters
        ----------
        repo_id : str
            A Hugging Face repository, such as
            ``"local/repo-name"``.
        data_dir : str | Path | None
            Path where data should be saved on disk. By default, files are
            saved in the standard nidl cache directory.
            Default None.
        recorded_instances : dict[str, Any] | None
            Dictionary of instances created inside a
            :meth:`ConfigLoader.record` context manager (or decorated with
            :meth:`ConfigLoader.record_config`).
            Default None.

        Returns
        -------
        cfg_file : Path
            Path to the saved configuration file.
        weights_file : Path
            Path to the saved estimator weights.

        Notes
        -----
        Force any parameters except ``repo_id`` to be passed by name (keyword)
        rather than by position.
        """
        check_is_fitted(self)

        data_dir = get_cache_dir(data_dir)

        local_dir = Path(data_dir) / repo_id.replace("/", "_")
        if local_dir.is_dir():
            rank_zero_info(
                f"WARNING: Save directory '{local_dir}' exists."
            )
        local_dir.mkdir(parents=True, exist_ok=True)

        weigths_file = local_dir / self.WEIGHTS_NAME
        Weights.save_pretrained(self, weigths_file)

        _instances = {
            "estimator": self,
        }
        _instances.update(recorded_instances or {})
        cfg_dict, valid_keys = {}, []
        for key, inst in _instances.items():
            if key not in ConfigLoader.ALLOWED_KEYS:
                rank_zero_info(
                    f"WARNING: Ignoring item '{key}' (allowed: "
                    f"{ConfigLoader.ALLOWED_KEYS})"
                )
                continue
            if not hasattr(inst, "_init_args"):
                cls_name = type(inst).__name__
                rank_zero_info(
                    f"WARNING: Instance of '{cls_name}' has no attribute "
                    "'_init_args'. Consider creating instance inside a "
                    "`ConfigLoader.record` context manager. Skipping "
                    "configuration recording for this instance."
                )
            valid_keys.append(key)
            cfg_dict[f"{key}"] = getattr(inst, "_init_args", None)

        cfg_file = local_dir / self.CONFIG_NAME
        ConfigLoader.save(cfg_dict, cfg_file)

        return cfg_file, weigths_file

    def push_pretrained(
            self,
            repo_id: str,
            *,
            recorded_instances: dict[str, Any] | None = None,
            commit_message: str | None = None,
            revision: str | None = None,
            token: str | bool | None = None,
            private: bool = False,
        ) -> None:
        """
        Save the estimator in the nidl format to a Hub repository.

        Parameters
        ----------
        repo_id : str
            Target repository identifier on the Hub (e.g.,
            ``"username/repo-name"``).
        recorded_instances : dict[str, Any] | None
            Dictionary of instances created inside a
            :meth:`ConfigLoader.record` context manager (or decorated with
            :meth:`ConfigLoader.record_config`).
        commit_message : str | None
            Custom commit message for the upload. If ``None``, a default
            message is generated automatically.
            Default None.
        revision : str | None
            Git branch or revision name to push the file to (e.g., ``"main"``).
        token : str | bool | None
            Hugging Face authentication token. If ``None``, uses the locally
            saved token (via ``huggingface-cli login`` or ``HF_TOKEN`` env
            variable).
            Default None.
        private : bool
            Whether the repository should be private if created by this
            function.
            Default False.

        Notes
        -----
        Force any parameters except ``repo_id`` to be passed by name (keyword)
        rather than by position.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            for local_file in self.save_pretrained(
                    repo_id=repo_id,
                    data_dir=tmp_dir,
                    recorded_instances=recorded_instances,
                ):
                push_file(
                    local_file=local_file,
                    repo_id=repo_id,
                    commit_message=commit_message,
                    revision=revision,
                    token=token,
                    private=private,
                )


class Weights:
    """
    Retrieve, apply, and save model weights or PyTorch Lightning checkpoints.

    Parameters
    ----------
    filepath : str | Path
        Path of the weight file on the local machine or relative path inside
        the Hugging Face repository.
    repo_id : str | None
        Target repository identifier on the Hub (e.g.,
        ``"username/repo-name"``) used if ``filepath`` does not exist locally.
        Default None.
    revision : str | None
        Git branch, tag, or commit hash to fetch the file from (e.g.,
        ``"main"``).
    token : str | None, default None
        Hugging Face token, needed for gated repositories. If None, the
        locally saved token or the `HF_TOKEN` variable is used.
    local_files_only : bool, default=False
        Whether to only use the local Hugging Face cache.
    data_dir : str | Path | None
        Directory where downloaded files should be cached. Defaults to the
        standard nidle cache directory.
        Default None.

    Examples
    --------
    Save a PyTorch module's weights to a ``.safetensors`` file and reload them
    into a fresh model instance:

    >>> import tempfile
    >>> from pathlib import Path
    >>> import torch
    >>> import torch.nn as nn
    >>> from nidl.utils.hub import Weights
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     weight_file = Path(tmp) / "model.safetensors"
    ...     model = nn.Linear(4, 2)
    ...     Weights.save_pretrained(model, weight_file)
    ...     loader = Weights(filepath=weight_file)
    ...     reloaded = nn.Linear(4, 2)
    ...     loader.load_pretrained(reloaded)
    ...     torch.equal(model.weight, reloaded.weight)
    True
    """
    HF_URL = "https://huggingface.co"

    def __init__(
            self,
            filepath: str | Path,
            repo_id: str | None = None,
            revision: str | None = None,
            token: str | None = None,
            local_files_only: bool = False,
            data_dir: str | Path | None = None,
        ) -> None:
        self.data_dir = get_cache_dir(data_dir)

        self.weight_file = Path(filepath)
        if not self.weight_file.is_file():
            self.weight_file = fetch_file(
                repo_id,
                filepath,
                revision=revision,
                token=token,
                data_dir=data_dir,
                local_files_only=local_files_only,
            )

    def load_checkpoint(
            self,
            model: pl.LightningModule,
            *,
            map_location: str | dict | torch.device = "cpu",
            **kwargs: Any,
        ) -> pl.LightningModule | None:
        """
        Load a PyTorch Lightning checkpoint into a LightningModule class.

        Parameters
        ----------
        model: pl.LightningModule
            A PyTorch Lightning module class exposing ``load_from_checkpoint``.
        map_location : str | dict | torch.device
            Device mapping used when loading the checkpoint.
        **kwargs: Any
            Extra keyword arguments passed to ``load_from_checkpoint``,
            such as overriding ``accelerator`` or ``devices```.

        Returns
        -------
        pl.LightningModule | None
            The instantiated LightningModule loaded from the checkpoint, or
            None if the provided file is not a ``.ckpt`` checkpoint.
        """
        if self.weight_file.suffix != ".ckpt":
            rank_zero_info(
                f"WARNING: The provided file ({self.weight_file}) does not "
                "seem to  be a Lightning's checkpoint file. To load weights "
                "directly, use `load_pretrained` instead."
            )
            return

        params_init = dict(
            kwargs,
            **({"devices": "auto"} if "devices" not in kwargs else {}),
            **({"accelerator": "auto"} if "accelerator" not in kwargs else {}),
        )
        return model.load_from_checkpoint(
            checkpoint_path=self.weight_file,
            map_location=map_location,
            **params_init,
        )

    def load_pretrained(
            self,
            model: torch.nn.Module,
            weights_only: bool = True,
            strict: bool = True,
        ) -> None:
        """
       Load serialized state dictionary weights into a PyTorch module.

        Parameters
        ----------
        model: torch.nn.Module
            Target PyTorch module into which weights will be loaded.
        weights_only: bool
            Whether ``torch.load`` should restrict unpickling to tensors,
            primitive types, and safe globals when loading non-safetensors
            files.
            Default True.
        strict : bool
            Whether the weights must match the estimator exactly. If False,
            a mismatch only raises a warning.
            Default True.
        """
        if self.weight_file is None:
            rank_zero_info(
                "WARNING: Define weight file location first!"
            )
            return
        if self.weight_file.suffix == ".ckpt":
            rank_zero_info(
                "WARNING: For pytorch_lightning checkpoints, use the method "
                "`load_checkpoint` instead."
            )
            return

        state_dict = (
            load_file(self.weight_file, device="cpu")
            if self.weight_file.suffix == ".safetensors"
            else torch.load(self.weight_file, weights_only=weights_only)
        )
        model.load_state_dict(
            state_dict,
            strict=strict,
        )

    @classmethod
    def save_pretrained(
            cls,
            model: torch.nn.Module,
            weight_file: str | Path,
        ) -> None:
        """
        Save a PyTorch module's state dictionary in ``safetensors`` format.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch module whose ``state_dict`` will be serialized.
        weight_file : str | Path
            Destination file path where the ``.safetensors`` weights will be
            written.
        """
        save_file(
            {
                name: value.detach().cpu().clone().contiguous()
                for name, value in model.state_dict().items()
            },
            weight_file,
            metadata={"format": "pt"},
        )


def fetch_file(
        repo_id: str,
        filename: str | Path,
        revision: str | None = None,
        token: str | bool | None = None,
        data_dir: str | Path | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        ignore_missing: bool = False,
    ) -> Path | None:
    """
    Download a file from the Hugging Face Hub and return its local path.

    Parameters
    ----------
    repo_id: str
        Target repository identifier on the Hub (e.g.,
        ``"username/repo-name"``).
    filename: str | Path
        Path of the file to download in the repository.
    revision: str | None
        Revision of the repository (a tag, or a commit hash).
        Default None.
    token: str | bool | None
        Hugging Face token, required for gated repositories. If None, the
        locally saved token or the `HF_TOKEN` variable is used.
        Default None.
    data_dir: str | Path | None
        Path where data should be downloaded. If None, the standard
        nidl cache is used.
        Default None.
    force_download: bool
        Whether the file should be downloaded even if it already exists in
        the local cache.
        Default False.
    local_files_only: bool
        Whether to only use the local cache, without network access.
        Default False.
    ignore_missing : bool
        If True, return None instead of raising an error when the file
        does not exist in the repository.
        Default False.

    Returns
    -------
    Path | None
        Local path to the downloaded file, or None if the file is missing
        and `ignore_missing` is True.
    """
    if ignore_missing:
        exists = file_exists(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type="model",
            token=token,
        )
        if not exists:
            return None

    data_dir = get_cache_dir(data_dir)

    local_dir = Path(data_dir) / repo_id.replace("/", "_")
    local_dir.mkdir(parents=True, exist_ok=True)

    local_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        repo_type="model",
        local_dir=local_dir,
        force_download=force_download,
        token=token,
        local_files_only=local_files_only,
    )

    return Path(local_file)


def push_file(
        local_file: str | Path,
        repo_id: str,
        path_in_repo: str | None = None,
        commit_message: str | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        private: bool = False,
    ) -> str:
    """
    Upload a local file to a Hugging Face Hub repository.

    Parameters
    ----------
    local_file : str | Path
        Path to the local file to be uploaded.
    repo_id : str
        Target repository identifier on the Hub (e.g.,
        ``"username/repo-name"``).
    path_in_repo : str | None
        Destination path inside the repository (using forward slashes ``/``).
        If ``None``, defaults to the local file's basename.
        Default None.
    commit_message : str | None
        Custom commit message for the upload. If ``None``, a default message
        is generated automatically.
        Default None.
    revision : str | None
        Git branch or revision name to push the file to (e.g., ``"main"``).
    token : str | bool | None
        Hugging Face authentication token. If ``None``, uses the locally saved
        token (via ``huggingface-cli login`` or ``HF_TOKEN`` env variable).
        Default None.
    private : bool
        Whether the repository should be private if created by this function.
        Default False.

    Returns
    -------
    str
        The HTTP URL of the uploaded file (commit URL) on the Hugging Face Hub.

    Raises
    ------
    FileNotFoundError
        If ``local_file`` does not exist or is not a file.
    huggingface_hub.utils.HfHubHTTPError
        If authentication fails or the user lacks write permissions on
        ``repo_id``.
    """
    local_file = Path(local_file)
    if not local_file.is_file():
        raise FileNotFoundError(f"Local file not found: '{local_file}'")

    target_path = (
        path_in_repo if path_in_repo is not None
        else local_file.name
    )
    message = commit_message or f"Upload {target_path}"

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )

    commit_info = api.upload_file(
        path_or_fileobj=local_file,
        path_in_repo=target_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=message,
        revision=revision,
    )

    return str(commit_info.commit_url)
