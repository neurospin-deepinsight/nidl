##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Configuration utilities.
"""

import importlib
import inspect
from contextlib import contextmanager
from functools import wraps
from importlib.metadata import (
    PackageNotFoundError,
)
from importlib.metadata import (
    version as get_pkg_version,
)
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

import torch
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import (
    DictConfig,
    OmegaConf,
)
from pytorch_lightning.utilities.rank_zero import rank_zero_info

T = TypeVar("T", bound=type[Any])


class ConfigLoader:
    """
    Load and instantiate Hydra configurations with version checks.

    The ``instantiate`` method works has follows:

    - Recursively validate ``_version_`` metadata against installed Python
      packages (or module ``__version__`` attributes) before delegating object
      creation to Hydra's ``instantiate``.
    - After Hydra instantiates the target block, all top-level keys of the
      resulting mapping are verified against ``cls.ALLOWED_KEYS`` before
      exposing them as attributes on a ``SimpleNamespace``.

    Examples
    --------
    >>> import torch
    >>> from importlib.metadata import version
    >>> from omegaconf import OmegaConf
    >>> from nidl.utils.config import ConfigLoader
    >>> cfg = OmegaConf.create({
    ...     "transform": {
    ...         "_target_": "torch.nn.Identity",
    ...     },
    ...     "estimator": {
    ...         "_target_": "torch.nn.Linear",
    ...         "in_features": 16,
    ...         "out_features": 8,
    ...     },
    ... })
    >>> obj = ConfigLoader.instantiate(cfg)
    >>> sorted(vars(obj).keys())
    ['estimator', 'transform']
    >>> isinstance(obj.transform, torch.nn.Identity)
    True
    >>> isinstance(obj.estimator, torch.nn.Linear)
    True
    """
    ALLOWED_KEYS = frozenset({
        "train_dataloader",
        "val_dataloader",
        "test_dataloader",
        "estimator",
        "weights",
        "transform",
    })

    @classmethod
    def instantiate(
            cls,
            cfg: str | Path | DictConfig,
            *args: Any,
            data_dir: str | Path | None = None,
            **kwargs: Any,
        ) -> SimpleNamespace:
        """
        Instantiate a Hydra configuration.

        Parameters
        ----------
        cfg : str | Path | DictConfig
            Root Hydra configuration or a nested sub-node to instantiate. Can
            be the path to the configuration file or a pre-loaded
            configuration.
        *args : Any
            Positional arguments forwarded directly to
            ``hydra.utils.instantiate``.
        data_dir : str | Path | None
            Path where data should be downloaded. By default, files are
            downloaded in the standard Hugging Face cache directory.
            Default None.
        **kwargs : Any
            Keyword arguments forwarded directly to
            ``hydra.utils.instantiate``.

        Returns
        -------
        SimpleNamespace
            The instantiated objects with keys in ``ALLOWED_KEYS``.

        Raises
        ------
        ValueError
            If unexpected keys are present in the returned object.
        """
        # Load config
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.load(cfg)

        # Copy the full root
        root = cfg._get_root()
        root_copy = OmegaConf.create(
            OmegaConf.to_container(root, resolve=False)
        )
        OmegaConf.set_struct(root_copy, False)

        # Replace 'data_dir: auto' in the tree
        if data_dir is not None and "data_dir" in root_copy:
            root_copy.data_dir = str(data_dir)

        # Resolve (${...}) befor isolating sub nodes
        OmegaConf.resolve(root_copy)

        # Validate _version_ across the full tree
        cls._validate_and_strip_versions(root_copy)

        # Instanciate objects
        instantiated = hydra_instantiate(root_copy, *args, **kwargs)

        # Check that instantiated keys belong to ``cls.ALLOWED_KEYS``
        keys = set(instantiated.keys())
        unexpected = keys - cls.ALLOWED_KEYS
        if unexpected:
            raise ValueError(
                "Unexpected key(s) in instantiated object: "
                f"{sorted(unexpected)}. Allowed keys: "
                f"{ sorted(cls.ALLOWED_KEYS)}."
            )

        return SimpleNamespace(**dict(instantiated))

    @staticmethod
    def load(
            cfg_file: str | Path,
            as_dict: bool = False,
        ) -> DictConfig | dict[str, Any]:
        """
        Load a configuration from a file.

        Parameters
        ----------
        cfg_file : str | Path
            Input config file.
        as_dict : bool
            Return a ductionaray rather than a `DictConfig` object.
            Default False.

        Returns
        -------
        DictConfig | dict[str, Any]
            Loaded config.
        """
        cfg = OmegaConf.load(cfg_file)
        if as_dict:
            cfg = OmegaConf.to_container(cfg, resolve=False)
        return cfg

    @staticmethod
    def save(
            cfg_dict: DictConfig | dict[str, Any],
            cfg_file: str | Path,
        ) -> None:
        """
        Save a configuration to disk.

        Parameters
        ----------
        cfg_dict : DictConfig | dict[str, Any]
            A dictionary containing ``_target_`` and initialization arguments.
        cfg_file : str | Path
            Generated configuration file.
        """
        cfg_str = OmegaConf.to_yaml(cfg_dict)
        Path(cfg_file).write_text(cfg_str, encoding="utf-8")

    @staticmethod
    def merge(
            *cfg_dicts: list[DictConfig | dict[str, Any]],
        ) -> DictConfig | dict[str, Any]:
        """
        Merge a list of previously created configs into a single one.

        Parameters
        ----------
        cfg_dicts : list[DictConfig | dict[str, Any]]
            Input configs

        Returns
        -------
        DictConfig | dict[str, Any]
            Merged configs.
        """
        seen_paths = set()
        collisions = set()
        for cfg in cfg_dicts:
            cfg_dict = (
                OmegaConf.to_container(cfg, resolve=False)
                if isinstance(cfg, DictConfig)
                else cfg
            )
            current_paths = ConfigLoader._get_paths(cfg_dict, target_depth=2)
            collisions.update(seen_paths & current_paths)
            seen_paths.update(current_paths)
        for path in sorted(collisions):
            rank_zero_info(
                "WARNING: Merge configuration collision detected at "
                f"'{path}'."
            )

        return OmegaConf.merge(*cfg_dicts)

    @staticmethod
    @contextmanager
    def record(
            *base_classes: type[Any],
            patch_existing: bool = True,
        ) -> None:
        """
        Temporarily record the first ``__init__`` for all existing and future
        subclasses of one or more ``base_classes``.

        Hooks into ``__init_subclass__`` of the target base class  while
        active and restores the original behavior upon exit.

        Parameters
        ----------
        base_classes : type
            One or more base classes whose hierarchies will be patched (e.g.,
            ``nn.Module`` or ``torch.utils.data.Dataset``).
        patch_existing : bool
            If True, also wraps existing subclasses during the context.
            Default True.

        Examples
        --------
        >>> import torch.nn as nn
        >>> from omegaconf import OmegaConf
        >>> from nidl.utils.config import ConfigLoader
        >>> cfg = OmegaConf.create({
        ...     "estimator": {
        ...         "_target_": "torch.nn.Linear",
        ...         "in_features": 16,
        ...         "out_features": 8,
        ...     },
        ... })
        >>> with ConfigLoader.record(nn.Module, ):
        ...     model = ConfigLoader.instantiate(cfg).estimator
        >>> model_cfg = (model._init_args
        >>> model_cfg["_target_"]
        'torch.nn.modules.linear.Linear'
        """
        saved_init_subclasses = {}
        saved_inits = {}

        def _wrap_cls(cls: type[Any]) -> None:
            if "__init__" in cls.__dict__ and cls not in saved_inits:
                orig_init = cls.__dict__["__init__"]
                try:
                    ConfigLoader.record_config(cls)
                    saved_inits[cls] = orig_init
                except (ValueError, TypeError):
                    pass

        def _patch_hierarchy(parent: type[Any]) -> None:
            for subcls in parent.__subclasses__():
                _wrap_cls(subcls)
                _patch_hierarchy(subcls)


        for base_cls in base_classes:
            # Skip duplicate base classes if passed twice
            if base_cls in saved_init_subclasses:
                continue

            _wrap_cls(base_cls)

            had_own = "__init_subclass__" in base_cls.__dict__
            orig_method = base_cls.__dict__.get("__init_subclass__")
            saved_init_subclasses[base_cls] = (had_own, orig_method)

            @classmethod
            def _auto_record_init_subclass(
                    cls: type[Any],
                    _base: type[Any] = base_cls,
                    _orig: Any = orig_method,
                    **kwargs: Any,
                ) -> None:
                if _orig is not None:
                    _orig.__func__(cls, **kwargs)
                else:
                    super(_base, cls).__init_subclass__(**kwargs)
                _wrap_cls(cls)

            base_cls.__init_subclass__ = _auto_record_init_subclass
            _patch_hierarchy(base_cls)

        base_cls.__init_subclass__ = _auto_record_init_subclass
        if patch_existing:
            _patch_hierarchy(base_cls)

        try:
            yield
        finally:
            # Restore __init_subclass__ for every patched base class
            for base_cls, state in saved_init_subclasses.items():
                had_own, orig_method = state
                if had_own:
                    base_cls.__init_subclass__ = orig_method
                else:
                    del base_cls.__init_subclass__

            # Restore original __init__ for every wrapped subclass
            for subcls, orig_init in saved_inits.items():
                subcls.__init__ = orig_init

    @staticmethod
    def record_config(
            target_cls: T,
        ) -> T:
        """
        Decorate a class to capture its outermost ``__init__`` arguments at
        instantiation.

        Captures positional, keyword, and default arguments passed to the first
        (outermost) ``__init__`` call, eagerly converting any nested objects
        that already have ``_init_args`` into Hydra configuration dictionaries
        rather than storing live object references. The resulting mapping is
        stored in ``self._init_args`` after the original ``__init__``
        completes.

        Parameters
        ----------
        target_cls : type
            The class whose ``__init__`` method will be wrapped in place.

        Returns
        -------
        type
            The modified class ``target_cls`` with argument recording enabled
            on instantiation.

        Attributes
        ----------
        _init_args : dict[str, Any]
            Injected directly into each instance's ``__dict__`` after
            initialization. Maps parameter names (excluding ``self``) to their
            normalized values or Hydra configuration dictionaries.

        Notes
        -----
        * Only wraps ``__init__`` if it is defined directly in
          ``target_cls.__dict__`` to avoid mutating parent class constructors.
        * Uses an instance-level re-entrancy counter (``_init_depth``) so
          nested ``super().__init__()`` calls do not overwrite the outer
          arguments.
        * Writes directly to ``self.__dict__`` to bypass custom ``__setattr__``
          hooks (such as ``torch.nn.Module`` submodule registration).

        Examples
        --------
        >>> from nidl.utils.config import ConfigLoader
        >>> @ConfigLoader.record_config
        ... class Backbone:
        ...     def __init__(self, hidden_dim: int = 128) -> None:
        ...         self.hidden_dim = hidden_dim
        >>> @ConfigLoader.record_config
        ... class Model:
        ...     def __init__(self, enc: Backbone, lr: float = 0.1) -> None:
        ...         self.enc = enc
        ...         self.lr = lr
        >>> m = Model(enc=Backbone(hidden_dim=6))
        >>> m._init_args["enc"]
        {'_target_': '__main__.Backbone', 'hidden_dim': 6}
        """
        orig_init = target_cls.__dict__.get("__init__")
        if orig_init is None or getattr(orig_init, "_is_recorded", False):
            return target_cls

        sig = inspect.signature(orig_init)

        @wraps(orig_init)
        def wrapped_init(
                self,
                *args: Any,
                **kwargs: Any,
            ) -> None:
            depth = self.__dict__.get("_init_depth", 0)
            self.__dict__["_init_depth"] = depth + 1

            try:
                orig_init(self, *args, **kwargs)

                if depth == 0:
                    bound = sig.bind(self, *args, **kwargs)
                    bound.apply_defaults()
                    dotpath = ConfigLoader._get_dotpath(type(self))
                    pkg_name = dotpath.split(".")[0]
                    installed_ver = ConfigLoader._get_installed_version(
                        pkg_name
                    )
                    self.__dict__["_init_args"] = {
                        "_target_": dotpath,
                        "_version_": installed_ver,
                        **{
                            key: ConfigLoader._extract_init_value(value)
                            for key, value in bound.arguments.items()
                            if key != "self"
                        },
                    }
                    if installed_ver is None:
                        del self.__dict__["_init_args"]["_version_"]
            finally:
                if depth == 0:
                    self.__dict__.pop("_init_depth", None)
                else:
                    self.__dict__["_init_depth"] = depth

        wrapped_init._is_recorded = True
        target_cls.__init__ = wrapped_init
        return target_cls

    @staticmethod
    def _get_installed_version(
            pkg_name: str,
        ) -> str | None:
        """
        Retrieve the installed version of a Python package or module.

        Queries ``importlib.metadata.version`` first. If the package is not
        installed as a distribution, it falls back to importing the module
        and reading its ``__version__`` attribute.

        Parameters
        ----------
        pkg_name : str
            Top-level distribution or importable module name (e.g., ``"torch"``
            or ``"my_project"``).

        Returns
        -------
        str | None
            The detected version string of the package or module, or ``None``
            if ``pkg_name`` is ``"__main__"``.

        Raises
        ------
        RuntimeError
            If the package is neither installed via package metadata nor
            importable with a ``__version__`` attribute.
        """
        if pkg_name == "__main__":
            return None
        try:
            return get_pkg_version(pkg_name)
        except PackageNotFoundError:
            try:
                module = importlib.import_module(pkg_name)
                return str(module.__version__)
            except (ImportError, AttributeError) as exc:
                raise RuntimeError(
                    f"Package '{pkg_name}' is not installed and has no "
                    "'__version__' attribute."
                ) from exc

    @classmethod
    def _validate_and_strip_versions(
            cls,
            node: Any,
        ) -> None:
        """
        Recursively validate and remove ``_version_`` keys in a configuration
        tree.

        Traverses OmegaConf dictionaries and lists in place. Whenever a
        mapping contains both ``_target_`` and ``_version_``, the expected
        version is popped and verified against the installed version of the
        target's top-level package.

        Parameters
        ----------
        node : Any
            Current configuration node (``DictConfig``, ``ListConfig``, or
            scalar) being traversed.

        Raises
        ------
        ValueError
            If the installed package version does not match ``_version_``.
        """
        if OmegaConf.is_dict(node):
            if "_target_" in node and "_version_" in node:
                expected_ver = str(node.pop("_version_"))
                pkg_name = str(node["_target_"]).split(".")[0]
                installed_ver = cls._get_installed_version(pkg_name)

                if installed_ver != expected_ver:
                    raise ValueError(
                        f"Version mismatch for '{pkg_name}': config expects "
                        f"{expected_ver}, but {installed_ver} is installed."
                    )

            for key in node:
                cls._validate_and_strip_versions(node[key])

        elif OmegaConf.is_list(node):
            for item in node:
                cls._validate_and_strip_versions(item)

    @staticmethod
    def _get_dotpath(
            target_cls: type[Any],
        ) -> str:
        """
        Return the full import dotpath of a class.

        Parameters
        ----------
        target_cls : type[Any]
            The class to get the import dotpath for.

        Returns
        -------
        str
            The full import dotpath in the form ``"<module>.<qualname>"``.

        Raises
        ------
        ValueError
            If ``cls`` is locally defined (i.e., ``"<locals>"`` appears in
            ``cls.__qualname__``).
        """
        if "<locals>" in target_cls.__qualname__:
            raise ValueError(
                "Cannot use locally defined class "
                f"'{target_cls.__qualname__}'."
            )
        return f"{target_cls.__module__}.{target_cls.__qualname__}"

    @staticmethod
    def _extract_init_value(val: Any) -> Any:
        """
        Recursively replace captured instances with their Hydra configuration
        dictionaries.

        If ``val`` exposes an ``_init_args`` attribute, it is converted into a
        dictionary containing ``_target_`` and its recorded initialization
        arguments instead of retaining a reference to the live object. Lists,
        tuples, dictionaries, and class types are traversed and converted
        recursively.

        Parameters
        ----------
        val : Any
            The argument value to inspect and normalize.

        Returns
        -------
        Any
            A Hydra-compatible configuration dictionary if ``val`` was captured
            or is a class type, a recursively normalized container, or the
            original value otherwise.
        """
        if hasattr(val, "_init_args"):
            dotpath = ConfigLoader._get_dotpath(type(val))
            pkg_name = dotpath.split(".")[0]
            installed_ver = ConfigLoader._get_installed_version(pkg_name)
            init_args = {
                "_target_": dotpath,
                "_version_": installed_ver,
                **{
                    _key: ConfigLoader._extract_init_value(_val)
                    for _key, _val in val._init_args.items()
                },
            }
            if installed_ver is None:
                del init_args["_version_"]
            return init_args
        if isinstance(val, dict):
            return {
                _key: ConfigLoader._extract_init_value(_val)
                for _key, _val in val.items()
            }
        if isinstance(val, (list, tuple)):
            return [
                ConfigLoader._extract_init_value(_val)
                for _val in val
            ]
        if isinstance(val, Path):
            return str(val)
        if isinstance(val, type):
            return {
                "_target_": "hydra.utils.get_class",
                "path": ConfigLoader._get_dotpath(val),
            }
        if isinstance(val, torch.dtype):
            return {
                "_target_": "hydra.utils.get_object",
                "path": str(val),
            }
        return val

    @staticmethod
    def _get_paths(
            cfg_dict: dict[str, Any],
            target_depth: int = 2,
            _prefix: str = "",
            _depth: int = 1,
        ) -> set[str]:
        """
        Extract dot-separated key paths located at a specific nesting depth.

        Parameters
        ----------
        cfg_dict : dict[str, Any]
            The input dictionary to extract key paths from.
        max_depth : int
            The maximum nesting depth to traverse.
            Default 2.
        _prefix : str
            Internal prefix accumulated during recursion.
            Defatult ''.
        _depth : int
            Internal current depth level.
            Default 1.

        Returns
        -------
        set[str]
            A set of dot-separated key paths at ``target_depth``.
        """
        paths = set()
        for key, value in cfg_dict.items():
            current_path = f"{_prefix}.{key}" if _prefix else str(key)
            if _depth == target_depth:
                paths.add(current_path)
            if isinstance(value, dict) and _depth < target_depth:
                paths.update(
                    ConfigLoader._get_paths(
                        value,
                        target_depth=target_depth,
                        _prefix=current_path,
                        _depth=_depth + 1,
                    )
                )
        return paths
