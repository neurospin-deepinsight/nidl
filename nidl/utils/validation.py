##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

from collections.abc import Sequence
from functools import update_wrapper, wraps
from inspect import isclass
from types import MethodType
from typing import Optional


def check_is_fitted(estimator, msg: Optional[str] = None):
    """Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises an Exception with the given message.

    Parameters
    ----------
    estimator: estimator instance
        Estimator instance for which the check is performed.
    msg: str, default=None
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        estimator."
        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.
        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    Raises
    ------
    TypeError
        If the estimator is a class or not an estimator instance.
    Exception
        If the fittted attribute is not found.
    """
    if isclass(estimator):
        raise TypeError(f"{estimator} is a class, not an instance.")
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this estimator."
        )
    if not hasattr(estimator, "fit"):
        raise TypeError(f"{estimator} is not an estimator instance.")
    if not hasattr(estimator, "fitted_") or not estimator.fitted_:
        raise RuntimeError(msg % {"name": type(estimator).__name__})


def _estimator_is(attrs: str or tuple[str]) -> bool:
    """Check if we can delegate a method to the underlying estimator."""
    if not isinstance(attrs, Sequence):
        attrs = (attrs,)
    return lambda estimator: (
        hasattr(estimator, "_estimator_type")
        and estimator._estimator_type in attrs
    )


class _AvailableIfDescriptor:
    """Implements a conditional property using the descriptor protocol.

    Using this class to create a decorator will raise an ``AttributeError``
    if check(self) returns a falsey value. Note that if check raises an error
    this will also result in hasattr returning false.

    See https://docs.python.org/3/howto/descriptor.html for an explanation of
    descriptors.
    """

    def __init__(self, fn, check, attribute_name):
        self.fn = fn
        self.check = check
        self.attribute_name = attribute_name

        # update the docstring of the descriptor
        update_wrapper(self, fn)

    def _check(self, obj, owner):
        attr_err_msg = (
            f"This {owner.__name__!r} has no attribute {self.attribute_name!r}"
        )
        try:
            check_result = self.check(obj)
        except Exception as e:
            raise AttributeError(attr_err_msg) from e

        if not check_result:
            raise AttributeError(attr_err_msg)

    def __get__(self, obj, owner=None):
        if obj is not None:
            # delegate only on instances, not the classes.
            # this is to allow access to the docstrings.
            self._check(obj, owner=owner)
            out = MethodType(self.fn, obj)

        else:
            # This makes it possible to use the decorated method as an
            # unbound method, for instance when monkeypatching.
            @wraps(self.fn)
            def out(*args, **kwargs):
                self._check(args[0], owner=owner)
                return self.fn(*args, **kwargs)

        return out


def available_if(check):
    """An attribute that is available only if check returns a truthy value.

    Parameters
    ----------
    check: callable
        When passed the object with the decorated method, this should return
        a truthy value if the attribute is available, and either return False
        or raise an AttributeError if not available.

    Returns
    -------
    callable
        Callable makes the decorated method available if `check` returns
        a truthy value, otherwise the decorated method is unavailable.

    Examples
    --------
    >>> from nidl.utils.validation import available_if
    >>> class HelloIfEven:
    ...    def __init__(self, x):
    ...        self.x = x
    ...
    ...    def _x_is_even(self):
    ...        return self.x % 2 == 0
    ...
    ...    @available_if(_x_is_even)
    ...    def say_hello(self):
    ...        print("Hello")
    ...
    >>> obj = HelloIfEven(1)
    >>> hasattr(obj, "say_hello")
    False
    >>> obj.x = 2
    >>> hasattr(obj, "say_hello")
    True
    >>> obj.say_hello()
    Hello
    """
    return lambda fn: _AvailableIfDescriptor(
        fn, check, attribute_name=fn.__name__
    )
