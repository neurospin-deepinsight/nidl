##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Configuration of nidl objects.

The configuration of an object is the dictionary of the parameters given to
its constructor, with a ``_target_`` key giving the import path of its class,
as in the `instantiate` function of `Hydra <https://hydra.cc>`_:

.. code-block:: yaml

    _target_: nidl.estimators.ssl.simclr.SimCLR
    encoder:
      _target_: nidl.backbones.volume.resnet3d.ResNet
      block:
        _target_: hydra.utils.get_class
        path: nidl.backbones.volume.resnet3d.BasicBlock
      layers: [2, 2, 2, 2]
    proj_input_dim: 512
    temperature: 0.1

The parameters are recorded when the object is created, by inheriting from
:class:`ConfigMixin`, and the object is rebuilt with :func:`instantiate`,
which relies on `hydra.utils.instantiate
<https://hydra.cc/docs/advanced/instantiate_objects/overview/>`_.

Configurations are downloaded from the Hugging Face Hub and are not trusted:
they can only build nidl classes and the containers of `torchvision
transforms <https://docs.pytorch.org/vision/stable/transforms.html>`_
(``Compose``, ``RandomApply``, ``RandomChoice`` and ``RandomOrder``), and only
give the classes of nidl, :mod:`torch.nn` and :mod:`torch.optim` as
parameters. Any other configuration is rejected before importing anything,
as well as the special keys of Hydra other than ``_target_`` (such as
``_partial_``) and its interpolations (``${...}``).
"""

import dataclasses
import functools
import inspect
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from hydra.utils import get_object
from hydra.utils import instantiate as hydra_instantiate

#: Prefix of the classes that a configuration can build.
BUILD_PREFIX = "nidl."

#: Prefixes of the classes that a configuration can give as parameters.
CLASS_PREFIXES = ("nidl.", "torch.nn.", "torch.optim.")

#: Target of a class given as a parameter, as in Hydra.
GET_CLASS = "hydra.utils.get_class"

#: Containers of torchvision transforms, and their saved parameters.
TORCHVISION_CONTAINERS = {
    "Compose": ("transforms",),
    "RandomApply": ("transforms", "p"),
    "RandomChoice": ("transforms", "p"),
    "RandomOrder": ("transforms",),
}

_PLAIN_TYPES = (type(None), bool, int, float, str)


class ConfigMixin:
    """Record the parameters given to the constructor of an object.

    The parameters are recorded when the object is created, before its
    ``__init__`` is called, so that :meth:`get_config` returns the
    configuration rebuilding it with :func:`instantiate`. Parameters taken
    by ``**kwargs`` are recorded with their own name, and the default value
    of the parameters that are not given is recorded too.

    A parameter is saved in the configuration if it is:

    - a number, a string, a boolean or None;
    - a list, a tuple or a dictionary with string keys of savable values;
    - a nidl object inheriting from :class:`ConfigMixin`, or a dataclass of
      nidl, saved as its own configuration;
    - a class, saved as its import path;
    - a container of torchvision transforms (``Compose``, ``RandomApply``,
      ``RandomChoice`` or ``RandomOrder``) of savable transforms.

    Any other parameter, such as a :class:`torch.nn.Module` that is not a
    nidl object, an optimizer instance or a function, cannot be saved and
    :meth:`get_config` raises an error, except if it is the default value of
    the parameter: then it is left out of the configuration, as the
    constructor sets it back.

    Attributes
    ----------
    _config_exclude : frozenset of str
        Names of the parameters left out of the configuration, because they
        depend on the data or the machine rather than on the object, such as
        the orientation of the input volumes. They must be given again when
        the object is rebuilt.

    Notes
    -----
    The objects copied with :mod:`copy` or :mod:`pickle` keep their
    configuration.

    Examples
    --------
    >>> from nidl.backbones.volume import resnet18
    >>> encoder = resnet18(n_embedding=128)
    >>> config = encoder.get_config()
    >>> config["_target_"]
    'nidl.backbones.volume.resnet3d.ResNet'
    >>> config["n_embedding"]
    128
    """

    _config_exclude: ClassVar[frozenset] = frozenset()

    def __new__(cls, *args: Any, **kwargs: Any):
        instance = super().__new__(cls)
        signature = _init_signature(cls)
        try:
            bound = signature.bind(None, *args, **kwargs)
        except TypeError:
            # `copy` and `pickle` create the object without parameters, then
            # restore its attributes, its configuration included.
            return instance
        bound.apply_defaults()
        params = {}
        for name, value in list(bound.arguments.items())[1:]:
            kind = signature.parameters[name].kind
            if kind is inspect.Parameter.VAR_KEYWORD:
                params.update(value)
            elif kind is inspect.Parameter.VAR_POSITIONAL:
                if value:  # cannot be given by name: not recorded
                    return instance
            else:
                params[name] = value
        # Bypass `__setattr__`, such as the one of torch.nn.Module.
        instance.__dict__["_init_params"] = params
        return instance

    def get_config(self) -> dict[str, Any]:
        """Return the configuration of this object.

        Returns
        -------
        config : dict
            The import path of the class of this object, under the
            ``_target_`` key, and the parameters given to its constructor.
            They only contain numbers, strings, booleans, None, lists and
            dictionaries.

        Raises
        ------
        TypeError
            If a parameter cannot be saved.
        """
        return get_config(self)

    def _config_params(self) -> dict[str, Any]:
        """Return the parameters saved in the configuration."""
        params = self.__dict__.get("_init_params")
        if params is None:
            raise TypeError(
                f"The parameters of this {type(self).__name__} were not "
                "recorded at its creation."
            )
        defaults = {
            name: parameter.default
            for name, parameter in _init_signature(
                type(self)
            ).parameters.items()
        }
        return {
            name: value
            for name, value in params.items()
            if name not in self._config_exclude
            and not (
                value is defaults.get(name, inspect.Parameter.empty)
                and not isinstance(value, _PLAIN_TYPES)
            )
        }


def get_config(obj: Any) -> dict[str, Any]:
    """Return the configuration of an object.

    Parameters
    ----------
    obj : object
        A nidl object inheriting from :class:`ConfigMixin`, or a container
        of torchvision transforms (``Compose``, ``RandomApply``,
        ``RandomChoice`` or ``RandomOrder``).

    Returns
    -------
    config : dict
        The import path of the class of the object, under the ``_target_``
        key, and the parameters given to its constructor.

    Raises
    ------
    TypeError
        If the object, or one of its parameters, cannot be saved.

    Examples
    --------
    >>> from torchvision.transforms import Compose
    >>> from nidl.transforms.volume.preprocessing import (
    ...     CropOrPad,
    ...     ZNormalization,
    ... )
    >>> from nidl.utils.config import get_config
    >>> config = get_config(Compose([CropOrPad(128), ZNormalization()]))
    >>> config["_target_"]
    'torchvision.transforms.Compose'
    >>> [t["_target_"] for t in config["transforms"]]
    ['nidl.transforms.volume.preprocessing.spatial.crop_or_pad.CropOrPad', \
'nidl.transforms.volume.preprocessing.intensity.z_normalization.ZNormalization']
    """
    if isinstance(obj, ConfigMixin):
        name = type(obj).__name__
        params = obj._config_params()
        return {
            "_target_": _class_path(type(obj)),
            **{
                key: _encode(value, f"{name}.{key}")
                for key, value in params.items()
            },
        }
    container = _torchvision_container(type(obj))
    if container is not None:
        path, params = container
        name = type(obj).__name__
        return {
            "_target_": path,
            **{
                key: _encode(
                    list(getattr(obj, key))
                    if key == "transforms"
                    else getattr(obj, key),
                    f"{name}.{key}",
                )
                for key in params
            },
        }
    raise TypeError(
        f"{type(obj).__name__} has no configuration: only the nidl objects "
        "and the containers of torchvision transforms can be saved."
    )


def check_config(config: Any) -> None:
    """Check that a configuration only builds allowed classes.

    The classes are checked before anything is built or imported, except
    the modules of nidl: a configuration downloaded from the Hugging Face Hub
    cannot run arbitrary code.

    Parameters
    ----------
    config : object
        A configuration, or any value of a configuration.

    Raises
    ------
    ValueError
        If the configuration builds, or gives as a parameter, a class that
        is not allowed, uses a special key of Hydra other than ``_target_``,
        or an interpolation.
    """
    if isinstance(config, list):
        for value in config:
            check_config(value)
        return
    if isinstance(config, str) and "${" in config:
        raise ValueError(
            f"Interpolations are not allowed in a configuration: {config!r}."
        )
    if not isinstance(config, dict):
        return
    if any(not isinstance(key, str) for key in config):
        raise ValueError("The keys of a configuration must be strings.")
    special = sorted(
        key for key in config if key.startswith("_") and key != "_target_"
    )
    if special:
        raise ValueError(
            f"Special keys {special} are not allowed in a configuration."
        )
    target = config.get("_target_")
    if target is None:
        pass
    elif not isinstance(target, str):
        raise ValueError(f"Invalid target {target!r}.")
    elif target == GET_CLASS:
        path = config.get("path")
        if set(config) != {"_target_", "path"} or not isinstance(path, str):
            raise ValueError(
                f"A class is given as {{'_target_': '{GET_CLASS}', 'path': "
                f"...}}, got {config}."
            )
        if not _is_allowed_class(path, CLASS_PREFIXES):
            raise ValueError(
                f"Class '{path}' cannot be given as a parameter: only the "
                f"classes of {', '.join(p[:-1] for p in CLASS_PREFIXES)} are "
                "allowed."
            )
    elif _torchvision_target(target) is not None:
        allowed = {"_target_", *_torchvision_target(target)}
        if not set(config) <= allowed:
            raise ValueError(
                f"Unexpected parameters {sorted(set(config) - allowed)} for "
                f"'{target}'."
            )
    elif not _is_allowed_class(
        target, (BUILD_PREFIX,), buildable=True
    ):
        raise ValueError(
            f"'{target}' cannot be built from a configuration: only the "
            "classes of nidl with a configuration and the containers of "
            "torchvision transforms are allowed."
        )
    for key, value in config.items():
        if key != "_target_":
            check_config(value)


def instantiate(config: dict[str, Any], **kwargs: Any) -> Any:
    """Build an object from its configuration.

    The configuration is checked with :func:`check_config`, then the
    parameters it saves are built with `hydra.utils.instantiate
    <https://hydra.cc/docs/advanced/instantiate_objects/overview/>`_, and
    given, with `kwargs`, to the class of the object.

    Parameters
    ----------
    config : dict
        The configuration, as returned by :func:`get_config`.
    **kwargs : dict
        Parameters of the object overriding, or completing, the ones of the
        configuration. They are given as is, without copy: they can be any
        Python object.

    Returns
    -------
    obj : object
        The new object.

    Raises
    ------
    ValueError
        If the configuration builds a class that is not allowed (see
        :func:`check_config`), or if a parameter without default value is
        neither in the configuration nor in `kwargs`.

    Examples
    --------
    >>> from nidl.transforms.volume.preprocessing import ZNormalization
    >>> from nidl.utils.config import instantiate
    >>> config = ZNormalization(eps=1e-6).get_config()
    >>> transform = instantiate(config)
    >>> type(transform).__name__
    'ZNormalization'
    """
    if not isinstance(config, dict) or "_target_" not in config:
        raise ValueError("A configuration is a dictionary with a '_target_'.")
    check_config(config)
    target = _resolve(config["_target_"])
    # The overrides are not given to Hydra, which would merge a dictionary
    # with the saved one instead of replacing it.
    params = hydra_instantiate(
        {key: value for key, value in config.items() if key != "_target_"},
        _convert_="all",
    )
    params.update(kwargs)
    parameters = list(_init_signature(target).parameters.items())[1:]  # self
    missing = [
        name
        for name, parameter in parameters
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and name not in params
    ]
    if missing:
        raise ValueError(
            f"{target.__name__} needs the parameters {missing}, which are not "
            "saved in its configuration: give them as keyword arguments."
        )
    return target(**params)


def resolve_target(config: dict[str, Any]) -> type:
    """Return the class built by a configuration, without building it.

    Parameters
    ----------
    config : dict
        The configuration, checked with :func:`check_config`.

    Returns
    -------
    cls : type
        The class named by its ``_target_``.
    """
    check_config(config)
    return _resolve(config["_target_"])


def _encode(value: Any, where: str) -> Any:
    """Turn a parameter into a value of a configuration."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_encode(item, f"{where}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{where}: the keys of a dictionary must be str.")
        return {
            key: _encode(item, f"{where}.{key}") for key, item in value.items()
        }
    if isinstance(value, type):
        return {"_target_": GET_CLASS, "path": _class_path(value)}
    if isinstance(value, ConfigMixin) or _torchvision_container(type(value)):
        return get_config(value)
    if dataclasses.is_dataclass(value):
        return {
            "_target_": _class_path(type(value)),
            **{
                field.name: _encode(
                    getattr(value, field.name), f"{where}.{field.name}"
                )
                for field in dataclasses.fields(value)
                if field.init
            },
        }
    raise TypeError(
        f"{where} cannot be saved in a configuration: {value!r}. Give a nidl "
        "object, a class and its parameters, or plain values instead."
    )


def _class_path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _is_allowed_class(
    path: str, prefixes: tuple[str, ...], buildable: bool = False
) -> bool:
    """Whether `path` names a class defined in a module of `prefixes`.

    The path is checked before importing anything, and the class after, so
    that a path such as ``nidl.module.subprocess.Popen`` is rejected. A
    buildable class must have a configuration.
    """
    if not path.startswith(prefixes):
        return False
    try:
        cls = _resolve(path)
    except ValueError:
        return False
    if not isinstance(cls, type) or not f"{cls.__module__}.".startswith(
        prefixes
    ):
        return False
    return not buildable or (
        issubclass(cls, ConfigMixin) or dataclasses.is_dataclass(cls)
    )


def _resolve(path: str) -> Any:
    """Import an object from its import path."""
    try:
        return get_object(path)
    except (ImportError, ValueError) as e:
        raise ValueError(f"Cannot import '{path}'.") from e


@functools.cache
def _init_signature(cls: type) -> inspect.Signature:
    return inspect.signature(cls.__init__)


@functools.cache
def _torchvision_containers() -> dict[type, tuple[str, tuple[str, ...]]]:
    import torchvision.transforms as transforms_v1
    from torchvision.transforms import v2 as transforms_v2

    containers = {}
    for module, prefix in (
        (transforms_v1, "torchvision.transforms"),
        (transforms_v2, "torchvision.transforms.v2"),
    ):
        for name, params in TORCHVISION_CONTAINERS.items():
            containers[getattr(module, name)] = (f"{prefix}.{name}", params)
    return containers


def _torchvision_container(cls: type):
    """Return the import path and parameters of a torchvision container."""
    return _torchvision_containers().get(cls)


def _torchvision_target(target: str):
    """Return the parameters of a torchvision container from its path."""
    for path, params in _torchvision_containers().values():
        if path == target:
            return params
    return None


def is_torchvision_container(cls: type) -> bool:
    """Whether a class is a container of torchvision transforms.

    Parameters
    ----------
    cls : type
        The class.

    Returns
    -------
    bool
        Whether `cls` is ``Compose``, ``RandomApply``, ``RandomChoice`` or
        ``RandomOrder``, from ``torchvision.transforms`` or
        ``torchvision.transforms.v2``.
    """
    return _torchvision_container(cls) is not None
