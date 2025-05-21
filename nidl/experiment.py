##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import collections
import copy
import importlib
import inspect
import itertools
import os
import warnings
from pprint import pprint
from typing import Optional

import toml

from .utils import Bunch, print_multicolor

SECTIONS = ("project", "global", "import",
            "scaler", "transform", "compose", "augmentation", "dataset",
            "dataloader", "model", "weights", "loss", "optimizer",
            "scheduler",
            "training",
            "environments")


def fetch_experiment(
        expfile: str,
        selector: Optional[tuple[str]] = None,
        cv: Optional[tuple[str]] = None,
        logdir: Optional[str] = None,
        verbose: int = 0):
    """ Fetch an experiement from an input configuration file.

    Allowed keys are:

    - project: define here some usefull information about your experiment such
      as the 'name', 'author', 'date'...
    - global: define here global variables that can be reused in other
      sections.
    - import: define here import that can be reused in other sections with the
      'auto' mode (see desciption below).
    - scaler: dl interface
    - transform: dl interfaces
    - compose: dl interfaces
    - augmentation: dl interface
    - dataset: dl interface
    - dataloader: dl interface
    - model: dl interface
    - weights: dl interface
    - loss: dl interface
    - optimizer: dl interface
    - scheduler: dl interface
    - training: define here training settings.
    - environements: define here the interface to load in order to fullfil your
      needs and the constraint impose by the 'interface_occurrences'
      parameter (see desciption below).

    Interface definition:

    - the 'interface' key contains a name that specifies what class to import
      in absolute terms.
    - the 'interface_version' key contains the expected version of the loaded
      interface. The '__version__' module parameter is checked if available
      and a warning is displayed if is mismatched is detected or the version
      connot be checked.
    - other key are the interface parameters.
    - dynamic parameters can be defined by specifying where this parameter
      can be find a previously loaded interface, i.e., 'auto|<name>.<param>'.
      Note that the order is important here.
    - cross validation can be defined by specifying a list of values, i.e.
      'cv|[1, 2, 3]'. This will automatically instanciate multiple interface,
      one for each input setting.

    The codes works as follows:

    - multiple interfaces of the same type can be returned.
    - the different choices must be described in an 'environments' section.
    - the output name will be prefixed by the environment name.
    - use the selector input parameters to filter the available interfaces in
      the input configuration file.

    How to define multiple building blocks:

    - the construction is hierarchic, i.e. child building blocks inherit
      the properties of the parent building block.
    - a child building block name contains the parent name as a prefix and use
      the '.' separator.

    The weights section special case:

    - model names specified in the form `hf-hub:path/architecture_name@revision`
      will be loaded from Hugging Face hub.
    - model names specifid with a path will be loaded from the local machine.

    Parameters
    ----------
    expfile: str
        the experimental design file.
    selector: tuple of str, default=None
        if multiple interface of the same type are defined, this parameter
        allows you to select the appropriate environements.
    cv: tuple of str, default=None
        if a cross validation scheme is defined, this parameter allows you to
        select only some interfaces (i.e., the best set of hyperparapeters).
    logdir: str, defautl=None
        allows you to save a couple of information about the loaded interface:
        for the moment only the source code of each interface.
    verbose: int, default=0
        enable verbose output.

    Returns
    -------
    data: Bunch
        dictionaray-like object containing the experiment building blocks.
    """
    if logdir is not None:
        assert os.path.isdir(logdir), "Please create the log directory!"
    config = toml.load(expfile, _dict=collections.OrderedDict)
    for key in config:
        assert key in SECTIONS, f"Unexpected section '{key}'!"
    settings = {key: config.pop(key) if key in config else None
                for key in ["project", "import", "global", "environments"]}
    selector = selector or []
    for key in selector:
        assert key in settings["environments"], (
            f"Unexpected environment '{key}'!")
    config_env = get_env(settings["global"], settings["import"])
    config = filter_config(config, settings["environments"], selector)
    if verbose > 0:
        print(f"[{print_multicolor('Configuration', display=False)}]")
        pprint(config)
    interfaces = {}
    cv_interfaces = [name.split("_")[0] for name in cv if cv is not None]
    for key, params in config.items():
        name = params.pop("interface") if "interface" in params else None
        if name is None:
            raise ValueError(f"No interface defined for '{key}'!")
        version = (params.pop("interface_version")
                   if "interface_version" in params else None)
        if "interface_occurrences" in params:
            params.pop("interface_occurrences")
        params, param_sets = update_params(interfaces, params, config_env)
        is_cv = (len(params) > 1)
        for _idx, _params in enumerate(params):
            if verbose > 0:
                print(f"\n[{print_multicolor('Loading', display=False)}] "
                      "{name}..."
                      f"\nParameters\n{'-'*10}")
                pprint(dict(_params))
            _key = f"{key}_{_idx}" if is_cv else key
            if (not is_cv or cv is None or key not in cv_interfaces or
                    _key in cv):
                interfaces[_key], code = load_interface(name, _params, version)
                if verbose > 0:
                    print(f"Interface\n{'-'*9}\n{interfaces[_key]}")
                if code is not None and logdir is not None:
                    logfile = os.path.join(logdir, name)
                    with open(logfile, "w") as of:
                        of.write(code)
        if is_cv:
            names = [f"{key}_{_idx}" for _idx in range(len(params))]
            _params = dict(zip(names, param_sets))
            interfaces.setdefault("grid", Bunch())[key] = load_interface(
                    "nidl.utils.Bunch", _params, None)[0]
    return Bunch(**interfaces)


def get_env(
        env: dict,
        modules: dict) -> dict:
    """ Dynamically update an environement.

    Parameters
    ----------
    env: dict
        a environment to update.
    modules: dict
        some module to add in the current environment

    Returns
    -------
    updated_env: dict
        the updated environemt with the input modules imported.
    """
    updated_env = copy.copy(env or {})
    if modules is not None:
        for key, name in modules.items():
            if "." in name:
                module_name, object_name = name.rsplit(".", 1)
            else:
                module_name, object_name = name, None
            mod = importlib.import_module(module_name)
            if object_name is not None:
                updated_env[key] = getattr(mod, object_name)
            else:
                updated_env[key] = mod
    for key, val in updated_env.items():
        if isinstance(val, str) and val.startswith("auto|"):
            attr = val.split("|")[-1]
            try:
                updated_env[key] = eval(attr, globals(), updated_env)
            except Exception as exc:
                print(f"\n[{print_multicolor('Help', display=False)}]..."
                      f"\nEnvironment\n{'-'*11}")
                pprint(updated_env)
                raise ValueError(
                    f"Can't find the '{attr}' dynamic global argument. Please "
                    "check for a typo in your configuration file.") from exc
    return updated_env


def filter_config(
        config: dict,
        env: dict,
        selector: tuple[str]) -> dict:
    """ Filter configuration based on declared environements and user selector.

    Parameters
    ----------
    config: dict
        the current configuration.
    env: dict
        the declared environements.
    selector: tuple of str
        if multiple interface of the same type are defined, this parameter
        allows you to select the appropriate environements.

    Returns
    -------
    filter_conf: dict
        the filtered configuration.
    """
    selected_env = {}
    for env_name in selector:
        for key, val in env[env_name].items():
            if isinstance(val, list):
                selected_env.setdefault(key, []).extend(val)
            else:
                selected_env.setdefault(key, []).append(val)
    filter_config = collections.OrderedDict()
    for section, params in config.items():
        shared_params, multi_params = {}, []
        for name in params:
            if isinstance(params[name], collections.OrderedDict):
                assert section in selected_env, (
                    f"Multi-interface '{section}' environments not defined "
                    "properly!")
                if name in selected_env[section]:
                    multi_params.append((name, params[name]))
            else:
                shared_params[name] = params[name]
        n_envs = (1 if len(multi_params) == 0 else len(multi_params))
        multi_envs = (len(multi_params) > 0)
        if ("interface_occurrences" in params
                and params["interface_occurrences"] != n_envs):
            raise ValueError(
                f"The maximum occurence of the '{section}' interface is not "
                f"respected: {params['interface_occurrences']} vs. {n_envs}. "
                "Please update the loaded environments accordingly.")
        if multi_envs:
            for name, _params in multi_params:
                _params.update(shared_params)
                if n_envs > 1:
                    filter_config[f"{section}_{name}"] = _params
                else:
                    filter_config[section] = _params
        else:
           filter_config[section] = shared_params
    return filter_config


def update_params(
        interfaces: dict,
        params: dict,
        env: dict) -> dict:
    """ Replace auto and cv parameters.

    Parameters
    ----------
    interfaces: dict
        the currently loaded interfaces.
    params: dict
        the interface parameters.
    env: dict
        the local environment.

    Returns
    -------
    updated_params: list of dict
        the interface parameters with the auto attributes replaced in place. In
        case of cross validation a list of parameters is returned.
    param_sets: list of dict
        the cross validation parameter sets. None means no cross validation.
    """
    env.update(globals())
    grid_search_params = {}
    for key, val in params.items():
        if isinstance(val, str) and val.startswith(("auto|", "cv|")):
            attr = val.split("|")[-1]
            try:
                params[key] = eval(attr, interfaces, env)
            except Exception as exc:
                interfaces.pop("__builtins__")
                print(f"\n[{print_multicolor('Help', display=False)}]..."
                      f"\nEnvironment\n{'-'*11}")
                pprint(env)
                print(f"\Interfaces\n{'-'*10}")
                pprint(interfaces)
                raise ValueError(
                    f"Can't find the '{attr}' dynamic argument. Please check "
                    "for a typo in your configuration file.") from exc
            interfaces.pop("__builtins__")
        if isinstance(val, str) and val.startswith("cv|"):
            grid_search_params[key] = params[key]
    if len(grid_search_params) > 0:
        keys = grid_search_params.keys()
        param_sets = [
            dict(zip(keys, values))
            for values in itertools.product(*grid_search_params.values())]
        _params = []
        for cv_params in param_sets:
            _params.append(copy.deepcopy(params))
            _params[-1].update(cv_params)
        params = _params
    else:
        param_sets = None
        params = [params]
    return params, param_sets


def load_interface(
        name: str,
        params: dict,
        version: Optional[str]):
    """ Load an interface. 

    Parameters
    ----------
    name: str
        the interface name argument that specifies what class to
        import in absolute terms, i.e. 'my_module.my_class'.
    params: dict
        the interface parameters.
    version: str, default None
        the exppected modulee version.

    Returns
    -------
    cls: object
        a class object.
    code: str
        the code of the output class object, None in case of issue. 
    """
    module_name, class_name = name.rsplit(".", 1)
    root_module_name = module_name.split(".")[0]
    root_mod = importlib.import_module(root_module_name)
    if version is not None:
        mod_version = getattr(root_mod, "__version__", None)
        if mod_version is None:
            warnings.warn(
                f"The '{module_name}' module has no '__version__' parameter!",
                ImportWarning, stacklevel=2)
        elif mod_version != version:
            warnings.warn(
                f"The '{name}' interface has a different version!",
                ImportWarning, stacklevel=2)
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    assert inspect.isclass(cls), "An interface MUST be defined as a class!"
    try:
        code = inspect.getsource(cls)
    except Exception:
        warnings.warn(
                f"Impossible to retrieve the '{name}' source code!",
                ImportWarning, stacklevel=2)
        code = None
    return cls(**params), code
