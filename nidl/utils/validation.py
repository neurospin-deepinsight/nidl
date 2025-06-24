##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from collections.abc import Sequence
from functools import update_wrapper, wraps
from inspect import isclass
from types import MethodType
from typing import Optional
from sklearn.utils import validation
import torch

def check_is_fitted(
        estimator,
        msg: Optional[str] = None):
    """ Checks if the estimator is fitted by verifying the presence of
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
        raise Exception(msg % {"name": type(estimator).__name__})

def check_array(
        X,
        accept_sparse=False,
        *,
        accept_large_sparse=True,
        dtype="numeric",
        order=None,
        copy=False,
        force_writeable=False,
        force_all_finite="deprecated",
        ensure_all_finite=None,
        ensure_non_negative=False,
        ensure_2d=True,
        allow_nd=False,
        ensure_min_samples=1,
        ensure_min_features=1,
        estimator=None,
        input_name=""):
    """
    Ensure the input is converted to a NumPy array, compatible with scikit-learn. 

    Parameters
    ----------
    X: object
        Input object to check / convert.
    
        accept_sparse : str, bool or list/tuple of str, default=False
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    accept_large_sparse : bool, default=True
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse=False will cause it to be accepted
        only if its indices are stored with a 32-bit dtype.

        .. versionadded:: 0.20

    dtype : 'numeric', type, list of type or None, default='numeric'
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : {'F', 'C'} or None, default=None
        Whether an array will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.

    copy : bool, default=False
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_writeable : bool, default=False
        Whether to force the output array to be writeable. If True, the returned array
        is guaranteed to be writeable, which may require a copy. Otherwise the
        writeability of the input array is preserved.

        .. versionadded:: 1.6

    force_all_finite : bool or 'allow-nan', default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accepts np.inf, np.nan, pd.NA in array.
        - 'allow-nan': accepts only np.nan and pd.NA values in array. Values
          cannot be infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `np.nan`

        .. deprecated:: 1.6
           `force_all_finite` was renamed to `ensure_all_finite` and will be removed
           in 1.8.

    ensure_all_finite : bool or 'allow-nan', default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accepts np.inf, np.nan, pd.NA in array.
        - 'allow-nan': accepts only np.nan and pd.NA values in array. Values
          cannot be infinite.

        .. versionadded:: 1.6
           `force_all_finite` was renamed to `ensure_all_finite`.

    ensure_non_negative : bool, default=False
        Make sure the array has only non-negative values. If True, an array that
        contains negative values will raise a ValueError.

        .. versionadded:: 1.6

    ensure_2d : bool, default=True
        Whether to raise a value error if array is not 2D.

    allow_nd : bool, default=False
        Whether to allow array.ndim > 2.

    ensure_min_samples : int, default=1
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.

    ensure_min_features : int, default=1
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.

    estimator : str or estimator instance, default=None
        If passed, include the name of the estimator in warning messages.

    input_name : str, default=""
        The data name used to construct the error message. In particular
        if `input_name` is "X" and the data has NaN values and
        allow_nan is False, the error message will link to the imputer
        documentation.

        .. versionadded:: 1.1.0

    Returns
    -------
    array_converted : object
        The converted and validated array.
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()  # Convert Tensor to NumPy array
    
    # Validate and enforce NumPy format for scikit-learn compatibility
    return validation.check_array(
        X, accept_sparse=accept_sparse,
        accept_large_sparse=accept_large_sparse,
        dtype=dtype,
        order=order,
        copy=copy,
        force_writeable=force_writeable,
        force_all_finite=force_all_finite,
        ensure_all_finite=ensure_all_finite,
        ensure_non_negative=ensure_non_negative,
        ensure_2d=ensure_2d,
        allow_nd=allow_nd,
        ensure_min_samples=ensure_min_samples,
        ensure_min_features=ensure_min_features,
        estimator=estimator,
        input_name=input_name
        ) 


def _estimator_is(
        attrs: str or tuple[str]) -> bool:
    """ Check if we can delegate a method to the underlying estimator.
    """
    if not isinstance(attrs, Sequence):
        attrs = (attrs, )
    return lambda estimator: (
        hasattr(estimator, "_estimator_type") and
        estimator._estimator_type in attrs
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
        attr_err_msg = (f"This {owner.__name__!r} has no attribute "
                        f"{self.attribute_name!r}")
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
    """ An attribute that is available only if check returns a truthy value.

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
        fn, check, attribute_name=fn.__name__)
