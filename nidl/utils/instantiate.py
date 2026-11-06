##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Instantiating objects.
"""

import copy
from typing import (
    Any,
    TypeVar,
)

T = TypeVar("T")


def build_instance(
        obj: T | type[T],
        deepcopy: bool = False,
        expected_type: type[T] | None = None,
        *obj_args: Any,
        **obj_kwargs: Any,
    ) -> T | None:
    """
    Builds or returns an instance of any class based on the provided
    configuration.

    Parameters
    ----------
    obj : T | type[T] | None
        The object to build or return. It can be:
        - an already instantiated object, in which case it will
          be used as is.
        - a class type, in which case it will be instantiated with
          the provided `obj_kwargs`.
        - None, in which case None is returned.
    deepcopy : bool
        Whether to return a deep copy of the object. Useful to avoid
        side effects if the same instance is shared across multiple modules.
        Default False.
    expected_type : Optional[type[T]], default=None
        An optional class type to enforce strict type checking. If provided,
        `obj` must be an instance or a subclass of `expected_type`.
        Default None.
    *obj_args : Any
        The arguments to use when instantiating the object class.
        Ignored if `obj` is already an instantiated object.
    **obj_kwargs : Any
        The keyword arguments to use when instantiating the object class.
        Ignored if `obj` is already an instantiated object.

    Returns
    -------
    T | None
        The instantiated or verified object.

    Raises
    ------
    TypeError
        If `obj` does not match the `expected_type` or is not a valid
        class/instance.
    """
    # Nothing to build
    if obj is None:
        return None

    # Check strict type constraint if expected_type is provided
    if expected_type is not None:
        is_inst = isinstance(obj, expected_type)
        is_cls = isinstance(obj, type) and issubclass(obj, expected_type)
        if not (is_inst or is_cls):
            raise TypeError(
                "Expected an instance or subclass of "
                f"'{expected_type.__name__}', but got '{type(obj).__name__}'."
            )

    # Instantiate / copy
    if isinstance(obj, type):
        obj = obj(*obj_args, **obj_kwargs)
    elif deepcopy:
        obj = copy.deepcopy(obj)

    return obj
