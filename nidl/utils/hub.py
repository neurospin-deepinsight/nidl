##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Share nidl objects with the Hugging Face Hub.

Each kind of object is saved in its own file of a repository,
and loaded with the `from_pretrained` method of its class:

- an estimator, with :meth:`BaseEstimator.from_pretrained
  <nidl.utils.hub.HubMixin.from_pretrained>`, from ``config.yaml``
  and ``model.safetensors``;
- the preprocessing of its inputs, with :meth:`Transform.from_pretrained
  <nidl.utils.hub.HubMixin.from_pretrained>`, from
  ``preprocessor_config.yaml``.

The configuration files are described in :mod:`nidl.utils.config`.
"""

import contextlib
import tempfile
import warnings
from pathlib import Path
from typing import Any, ClassVar, Optional, Union

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from ..info import __version__
from .config import (
    ConfigMixin,
    check_config,
    get_config,
    instantiate,
    is_torchvision_container,
    resolve_target,
)
from .validation import get_cache_dir

#: Key of the version of nidl that saved a configuration file.
VERSION_KEY = "nidl_version"


class HubMixin(ConfigMixin):
    """Save, share and load nidl objects with the Hugging Face Hub.

    An object is saved in a local folder, or a Hugging Face repository, as:

    - a YAML configuration file, named after :attr:`CONFIG_NAME`, holding
      the version of nidl that saved it and, under the :attr:`CONFIG_KEY`
      key, the configuration of the object (see :class:`ConfigMixin
      <nidl.utils.config.ConfigMixin>`);
    - its weights, in :attr:`WEIGHTS_NAME`, if it has some.

    Several kinds of objects can be saved in the same repository, such as an
    estimator and the preprocessing of its inputs, as their files differ.

    The configuration of the object is built from the parameters given to
    its constructor. When saving, it is checked that it rebuilds the same
    object, so that a configuration downloaded from the Hub always gives
    the object that was saved.

    Attributes
    ----------
    CONFIG_NAME : str
        Name of the configuration file.
    CONFIG_KEY : str
        Key of the configuration of the object in this file.
    WEIGHTS_NAME : str or None
        Name of the file of the weights, or None if the object has none.
    """

    CONFIG_NAME: ClassVar[str] = "config.yaml"
    CONFIG_KEY: ClassVar[str] = "object"
    WEIGHTS_NAME: ClassVar[Optional[str]] = None

    @classmethod
    def from_pretrained(
        cls,
        repo_id: Union[str, Path],
        *,
        revision: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        local_files_only: bool = False,
        strict: bool = True,
        data_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ):
        """Build an object saved in a local folder or on the Hub.

        Parameters
        ----------
        repo_id : str or Path
            A Hugging Face repository, such as
            ``"nidl-neurospin/brainiac-vit-b"``, or a local folder, written
            by :meth:`save_pretrained`.
        revision : str or None, default=None
            Commit, branch or tag of the repository. By default, its last
            commit.
        token : str, bool or None, default=None
            Hugging Face token, needed for gated repositories. If None, the
            locally saved token or the `HF_TOKEN` variable is used.
        local_files_only : bool, default=False
            Whether to only use the files already downloaded.
        strict : bool, default=True
            Whether the weights must match the object exactly. If False, a
            mismatch only raises a warning. Ignored for objects without
            weights.
        data_dir : str, Path or None, default=None
            Folder where the files are downloaded. By default, the nidl
            cache directory (see the `NIDL_CACHE` variable).
        **kwargs : dict
            Parameters of the object overriding the ones of its
            configuration, such as the `devices` of an estimator, or
            completing them, such as the parameters that depend on your data.

        Returns
        -------
        obj : object
            The object. Calling this method on a base class, such as
            :class:`~nidl.transforms.transforms.Transform`, returns an
            instance of the saved subclass.

        Raises
        ------
        PermissionError
            If the repository is gated and access has not been granted.
        ValueError
            If the configuration is not in the nidl format, is not for this
            class, or builds a class that is not allowed (see
            :func:`~nidl.utils.config.check_config`).
        RuntimeError
            If `strict` and the weights do not match the object.
        """
        fetch_kwargs = {
            "token": token,
            "local_files_only": local_files_only,
            "data_dir": data_dir,
        }
        config_file = fetch_file(
            repo_id, cls.CONFIG_NAME, revision=revision, **fetch_kwargs
        )
        config = _read_config(config_file)
        node = config.get(cls.CONFIG_KEY) if isinstance(config, dict) else None
        if not isinstance(node, dict) or "_target_" not in node:
            raise ValueError(
                f"'{repo_id}' is not in the nidl format: its "
                f"{cls.CONFIG_NAME} has no '{cls.CONFIG_KEY}' entry."
            )
        _check_version(config.get(VERSION_KEY), repo_id)
        target = resolve_target(node)
        if not cls._accepts(target):
            hint = (
                f": use {target.__name__}.from_pretrained instead"
                if issubclass(target, HubMixin)
                else ""
            )
            raise ValueError(
                f"'{repo_id}' holds a {target.__name__}, not a "
                f"{cls.__name__}{hint}."
            )
        obj = instantiate(node, **kwargs)
        if isinstance(obj, HubMixin):
            obj._load_pretrained(
                config,
                repo_id,
                strict=strict,
                revision=revision,
                **fetch_kwargs,
            )
        return obj

    def save_pretrained(self, save_directory: Union[str, Path]) -> Path:
        """Save this object in a local folder.

        Parameters
        ----------
        save_directory : str or Path
            The folder, created if needed. Its files of the same name are
            replaced, the other ones are kept.

        Returns
        -------
        save_directory : Path
            The folder.

        Raises
        ------
        TypeError
            If a parameter of the object cannot be saved (see
            :class:`~nidl.utils.config.ConfigMixin`).
        ValueError
            If the object cannot be saved, for instance because its modules
            were changed after its creation, so that its configuration does
            not rebuild it.
        RuntimeError
            If the object is an estimator that is not fitted.
        """
        return save_pretrained(self, save_directory)

    def push_to_hub(
        self,
        repo_id: str,
        *,
        commit_message: Optional[str] = None,
        private: bool = False,
        revision: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """Save this object to a Hugging Face repository, in one commit.

        Parameters
        ----------
        repo_id : str
            The repository, such as ``"username/repo-name"``, created if
            needed. Its files of the same name are replaced, the other ones
            are kept.
        commit_message : str or None, default=None
            The message of the commit. By default, "Upload" and the name of
            the class.
        private : bool, default=False
            Whether a new repository is private.
        revision : str or None, default=None
            Branch to push to. By default, the main branch.
        token : str or None, default=None
            Hugging Face token. If None, the locally saved token or the
            `HF_TOKEN` variable is used.

        Returns
        -------
        commit : huggingface_hub.CommitInfo
            The commit, whose `oid` can be given as `revision` to
            :meth:`from_pretrained`.
        """
        return push_to_hub(
            self,
            repo_id,
            commit_message=commit_message,
            private=private,
            revision=revision,
            token=token,
        )

    @classmethod
    def _accepts(cls, target: type) -> bool:
        """Whether `from_pretrained` of this class can build `target`."""
        return issubclass(target, cls)

    def _check_saveable(self) -> None:
        """Raise an error if this object cannot be saved."""

    def _load_pretrained(
        self,
        config: dict[str, Any],
        repo_id: Union[str, Path],
        *,
        strict: bool,
        **fetch_kwargs: Any,
    ) -> None:
        """Load the weights of this object, built from `config`."""


def save_pretrained(obj: Any, save_directory: Union[str, Path]) -> Path:
    """Save a nidl object, or a container of torchvision transforms.

    The object is saved as by :meth:`HubMixin.save_pretrained`. A container
    of transforms (``Compose``, ``RandomApply``, ``RandomChoice`` or
    ``RandomOrder``, of ``torchvision.transforms`` or
    ``torchvision.transforms.v2``) is saved as a transform: it is loaded
    with :meth:`Transform.from_pretrained
    <nidl.utils.hub.HubMixin.from_pretrained>`.

    Parameters
    ----------
    obj : HubMixin or torchvision container
        The object.
    save_directory : str or Path
        The folder, created if needed. Its files of the same name are
        replaced, the other ones are kept.

    Returns
    -------
    save_directory : Path
        The folder.

    Raises
    ------
    TypeError
        If the object, or one of its parameters, cannot be saved.
    ValueError
        If the configuration of the object does not rebuild it.

    Examples
    --------
    >>> from torchvision.transforms import Compose
    >>> from nidl.transforms import Transform
    >>> from nidl.transforms.volume.preprocessing import (
    ...     CropOrPad,
    ...     ZNormalization,
    ... )
    >>> from nidl.utils.hub import save_pretrained
    >>> preprocessing = Compose([CropOrPad(128), ZNormalization()])
    >>> save_pretrained(preprocessing, "my_model")
    PosixPath('my_model')
    >>> preprocessing = Transform.from_pretrained("my_model")
    """
    hub_class = _hub_class(obj)
    if isinstance(obj, HubMixin):
        obj._check_saveable()
    config = get_config(obj)
    try:
        check_config(config)
    except ValueError as e:
        raise ValueError(
            f"This {type(obj).__name__} cannot be saved: {e}"
        ) from e
    _check_rebuild(obj, config)
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)
    if hub_class.WEIGHTS_NAME is not None:
        Weights.save_pretrained(obj, save_directory / hub_class.WEIGHTS_NAME)
    _write_config(
        {VERSION_KEY: __version__, hub_class.CONFIG_KEY: config},
        save_directory / hub_class.CONFIG_NAME,
    )
    return save_directory


def push_to_hub(
    obj: Any,
    repo_id: str,
    *,
    commit_message: Optional[str] = None,
    private: bool = False,
    revision: Optional[str] = None,
    token: Optional[str] = None,
):
    """Save a nidl object, or a container of torchvision transforms, to a
    Hugging Face repository, in one commit.

    The object is saved as by :func:`save_pretrained`.

    Parameters
    ----------
    obj : HubMixin or torchvision container
        The object.
    repo_id : str
        The repository, such as ``"username/repo-name"``, created if needed.
        Its files of the same name are replaced, the other ones are kept.
    commit_message : str or None, default=None
        The message of the commit. By default, "Upload" and the name of the
        class.
    private : bool, default=False
        Whether a new repository is private.
    revision : str or None, default=None
        Branch to push to. By default, the main branch.
    token : str or None, default=None
        Hugging Face token. If None, the locally saved token or the
        `HF_TOKEN` variable is used.

    Returns
    -------
    commit : huggingface_hub.CommitInfo
        The commit, whose `oid` can be given as `revision` to
        `from_pretrained`.
    """
    huggingface_hub = _import_hub_dependency("huggingface_hub")
    api = huggingface_hub.HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_pretrained(obj, tmp_dir)
        return api.upload_folder(
            repo_id=repo_id,
            folder_path=tmp_dir,
            commit_message=commit_message or f"Upload {type(obj).__name__}",
            revision=revision,
        )


class Weights:
    """
    Retrieve, apply, and save model weights or PyTorch Lightning checkpoints.

    Parameters
    ----------
    filepath : str or Path
        Path of the weight file on the local machine or relative path inside
        the Hugging Face repository.
    repo_id : str or None, default=None
        Target repository identifier on the Hub (e.g.,
        ``"username/repo-name"``) used if ``filepath`` does not exist locally.
    revision : str or None, default=None
        Git branch, tag, or commit hash to fetch the file from (e.g.,
        ``"main"``).
    token : str or None, default=None
        Hugging Face token, needed for gated repositories. If None, the
        locally saved token or the `HF_TOKEN` variable is used.
    local_files_only : bool, default=False
        Whether to only use the local Hugging Face cache.
    data_dir : str, Path or None, default=None
        Directory where downloaded files should be cached. By default, the
        nidl cache directory.

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

    def __init__(
        self,
        filepath: Union[str, Path],
        repo_id: Optional[str] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        local_files_only: bool = False,
        data_dir: Optional[Union[str, Path]] = None,
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
        map_location: Union[str, dict, torch.device] = "cpu",
        **kwargs: Any,
    ) -> Optional[pl.LightningModule]:
        """
        Load a PyTorch Lightning checkpoint into a LightningModule class.

        Parameters
        ----------
        model : pl.LightningModule
            A PyTorch Lightning module class exposing ``load_from_checkpoint``.
        map_location : str, dict or torch.device, default="cpu"
            Device mapping used when loading the checkpoint.
        **kwargs : Any
            Extra keyword arguments passed to ``load_from_checkpoint``,
            such as overriding ``accelerator`` or ``devices``.

        Returns
        -------
        pl.LightningModule or None
            The instantiated LightningModule loaded from the checkpoint, or
            None if the provided file is not a ``.ckpt`` checkpoint.
        """
        if self.weight_file.suffix != ".ckpt":
            rank_zero_info(
                f"WARNING: The provided file ({self.weight_file}) does not "
                "seem to be a Lightning's checkpoint file. To load weights "
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
        model : torch.nn.Module
            Target PyTorch module into which weights will be loaded.
        weights_only : bool, default=True
            Whether ``torch.load`` should restrict unpickling to tensors,
            primitive types, and safe globals when loading non-safetensors
            files.
        strict : bool, default=True
            Whether the weights must match the module exactly.
        """
        if self.weight_file.suffix == ".ckpt":
            rank_zero_info(
                "WARNING: For pytorch_lightning checkpoints, use the method "
                "`load_checkpoint` instead."
            )
            return

        state_dict = (
            load_safetensors(self.weight_file)
            if self.weight_file.suffix == ".safetensors"
            else torch.load(self.weight_file, weights_only=weights_only)
        )
        model.load_state_dict(state_dict, strict=strict)

    @classmethod
    def save_pretrained(
        cls,
        model: torch.nn.Module,
        weight_file: Union[str, Path],
    ) -> None:
        """
        Save a PyTorch module's state dictionary in ``safetensors`` format.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch module whose ``state_dict`` will be serialized.
        weight_file : str or Path
            Destination file path where the ``.safetensors`` weights will be
            written.
        """
        safetensors_torch = _import_hub_dependency("safetensors.torch")
        safetensors_torch.save_file(
            {
                name: value.detach().cpu().clone().contiguous()
                for name, value in model.state_dict().items()
            },
            str(weight_file),
            metadata={"format": "pt"},
        )


def fetch_file(
    repo_id: Union[str, Path],
    filename: Union[str, Path],
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    data_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
    local_files_only: bool = False,
) -> Path:
    """
    Return the local path of a file of a local folder or of a Hugging Face
    repository, downloaded if needed.

    Parameters
    ----------
    repo_id : str or Path
        A local folder, or a repository on the Hub (e.g.,
        ``"username/repo-name"``).
    filename : str or Path
        Path of the file in the folder or the repository.
    revision : str or None, default=None
        Revision of the repository (a branch, a tag, or a commit hash).
        Ignored for a local folder.
    token : str, bool or None, default=None
        Hugging Face token, required for gated repositories. If None, the
        locally saved token or the `HF_TOKEN` variable is used.
    data_dir : str, Path or None, default=None
        Path where the files of the repository are downloaded, in a folder
        named after it. By default, the nidl cache directory.
    force_download : bool, default=False
        Whether the file should be downloaded even if it already exists in
        the local cache.
    local_files_only : bool, default=False
        Whether to only use the local cache, without network access.

    Returns
    -------
    Path
        Local path to the file.

    Raises
    ------
    FileNotFoundError
        If the local folder has no such file.
    PermissionError
        If the repository is gated and access has not been granted.
    """
    folder = Path(repo_id)
    if folder.is_dir():
        if not (folder / filename).is_file():
            raise FileNotFoundError(f"No '{filename}' in '{folder}'.")
        return folder / filename

    huggingface_hub = _import_hub_dependency("huggingface_hub")
    from huggingface_hub.utils import GatedRepoError

    local_dir = get_cache_dir(data_dir) / str(repo_id).replace("/", "_")
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        local_file = huggingface_hub.hf_hub_download(
            repo_id=str(repo_id),
            filename=str(filename),
            revision=revision,
            repo_type="model",
            local_dir=local_dir,
            force_download=force_download,
            token=token,
            local_files_only=local_files_only,
        )
    except GatedRepoError as e:
        raise PermissionError(
            f"Access to the gated repository '{repo_id}' has not been "
            f"granted to this account. Request it on "
            f"https://huggingface.co/{repo_id}, then authenticate with "
            "`huggingface-cli login` or the `token` parameter."
        ) from e
    return Path(local_file)


def load_safetensors(path: Union[str, Path]) -> dict[str, torch.Tensor]:
    """Load a state dictionary saved in the safetensors format.

    Parameters
    ----------
    path : str or Path
        The ``.safetensors`` file.

    Returns
    -------
    state_dict : dict of str to torch.Tensor
        The tensors, on the CPU.
    """
    safetensors_torch = _import_hub_dependency("safetensors.torch")
    return safetensors_torch.load_file(str(path), device="cpu")


def _hub_class(obj: Any) -> type:
    """Return the class giving the file names of `obj`."""
    if isinstance(obj, HubMixin):
        return type(obj)
    if is_torchvision_container(type(obj)):
        from ..transforms import Transform

        return Transform
    raise TypeError(
        f"{type(obj).__name__} cannot be saved: only the nidl estimators, "
        "the nidl transforms and the containers of torchvision transforms "
        "can be."
    )


def _check_rebuild(obj: Any, config: dict[str, Any]) -> None:
    """Check that `config` rebuilds `obj`, without its weights."""
    # The parameters without default value left out of the configuration,
    # such as the orientation of the volumes, are given again.
    params = obj.__dict__.get("_init_params", {})
    overrides = {
        name: params[name]
        for name in getattr(obj, "_config_exclude", ())
        if name in params
    }
    is_module = isinstance(obj, torch.nn.Module)
    try:
        # Modules are built on the meta device, without allocating memory.
        with torch.device("meta") if is_module else contextlib.nullcontext():
            clone = instantiate(config, **overrides)
    except Exception:  # ruff: ignore[blind-except]
        # Some modules cannot be built on the meta device: retry normally.
        try:
            clone = instantiate(config, **overrides)
        except Exception as e:
            raise ValueError(
                f"This {type(obj).__name__} cannot be rebuilt from its "
                f"configuration: {e}"
            ) from e
    if get_config(clone) != config:
        raise ValueError(
            f"This {type(obj).__name__} cannot be rebuilt from its "
            "configuration: its parameters change when it is rebuilt."
        )
    if is_module:
        shapes = {k: tuple(v.shape) for k, v in obj.state_dict().items()}
        expected = {k: tuple(v.shape) for k, v in clone.state_dict().items()}
        if shapes != expected:
            differences = sorted(
                key
                for key in set(shapes) | set(expected)
                if shapes.get(key) != expected.get(key)
            )
            raise ValueError(
                f"The modules of this {type(obj).__name__} differ from the "
                "ones built from its parameters, for instance because they "
                "were changed after its creation: it cannot be rebuilt from "
                f"its configuration. Differences: {differences[:5]}."
            )


def _check_version(version: Optional[str], repo_id: Union[str, Path]) -> None:
    """Warn if `version` of nidl is more recent than the installed one."""
    if version is None:
        return
    from packaging.version import InvalidVersion, Version

    try:
        newer = Version(str(version)) > Version(__version__)
    except InvalidVersion:
        return
    if newer:
        warnings.warn(
            f"'{repo_id}' was saved with nidl {version}, more recent than "
            f"the installed version ({__version__}): update nidl if it "
            "cannot be loaded.",
            stacklevel=3,
        )


def _read_config(path: Union[str, Path]) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class _ConfigDumper(yaml.SafeDumper):
    """Write the lists of numbers or strings on one line."""


def _represent_list(dumper: yaml.SafeDumper, data: list) -> yaml.Node:
    flow = all(isinstance(value, (bool, int, float, str)) for value in data)
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=flow
    )


_ConfigDumper.add_representer(list, _represent_list)


def _write_config(config: dict[str, Any], path: Union[str, Path]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            config,
            f,
            Dumper=_ConfigDumper,
            sort_keys=False,
            default_flow_style=False,
        )


def _import_hub_dependency(module_name: str):
    """Import a dependency of the `hub` extra."""
    import importlib

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"'{module_name}' is needed to share models with the Hugging "
            "Face Hub. Please run 'pip install nidl[hub]' first."
        ) from e
