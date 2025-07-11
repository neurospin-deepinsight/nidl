"""Checks on regression variables. This is a re-implementation of scikit-learn
functions that does not require the latest numpy >= 2.0 version."""

import numpy as np
from sklearn.utils.validation import (
    _check_sample_weight,
    check_array,
    check_consistent_length,
)


def _check_reg_targets(
    y_true, y_pred, sample_weight, multioutput, dtype="numeric"
):
    """Check that y_true, y_pred and sample_weight belong to the same
    regression task.

    To reduce redundancy when calling `_find_matching_floating_dtype`,
    please use `_check_reg_targets_with_floating_dtype` instead.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,) or None
        Sample weights.

    multioutput : array-like or string in ['raw_values', uniform_average',
        'variance_weighted'] or None
        None is accepted due to backward compatibility of r2_score().

    dtype : str or list, default="numeric"
        the dtype argument passed to check_array.

    Returns
    -------
    type_true : one of {'continuous', continuous-multioutput'}
        The type of the true target data, as output by
        'utils.multiclass.type_of_target'.

    y_true : array-like of shape (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,) or None
        Sample weights.

    multioutput : array-like of shape (n_outputs) or string in ['raw_values',
        uniform_average', 'variance_weighted'] or None
        Custom output weights if ``multioutput`` is array-like or
        just the corresponding argument if ``multioutput`` is a
        correct keyword.
    """
    check_consistent_length(y_true, y_pred, sample_weight)
    y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
    y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)
    if sample_weight is not None:
        sample_weight = _check_sample_weight(
            sample_weight, y_true, dtype=dtype
        )

    if y_true.ndim == 1:
        y_true = np.reshape(y_true, (-1, 1))

    if y_pred.ndim == 1:
        y_pred = np.reshape(y_pred, (-1, 1))

    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError(
            f"y_true and y_pred have different number of output "
            f"({y_true.shape[1]}!={y_pred.shape[1]})"
        )

    n_outputs = y_true.shape[1]
    allowed_multioutput_str = (
        "raw_values",
        "uniform_average",
        "variance_weighted",
    )
    if isinstance(multioutput, str):
        if multioutput not in allowed_multioutput_str:
            raise ValueError(
                f"Allowed 'multioutput' string values are"
                f"{allowed_multioutput_str}."
                f"You provided multioutput={multioutput!r}"
            )
    elif multioutput is not None:
        multioutput = check_array(multioutput, ensure_2d=False)
        if n_outputs == 1:
            raise ValueError(
                "Custom weights are useful only in multi-output cases."
            )
        elif n_outputs != multioutput.shape[0]:
            raise ValueError(
                "There must be equally many custom weights "
                f"({multioutput.shape[0]}) as outputs ({n_outputs})."
            )
    y_type = "continuous" if n_outputs == 1 else "continuous-multioutput"

    return y_type, y_true, y_pred, sample_weight, multioutput


def _find_matching_floating_dtype(*arrays):
    """Find a suitable floating point dtype when computing with arrays.

    If any of the arrays are floating point, return the dtype with the highest
    precision by following official type promotion rules:

    https://data-apis.org/array-api/latest/API_specification/type_promotion.html

    If there are no floating point input arrays (all integral inputs for
    instance), return the default floating point dtype for the namespace.
    """
    dtyped_arrays = [np.asarray(a) for a in arrays if hasattr(a, "dtype")]
    floating_dtypes = [
        a.dtype for a in dtyped_arrays if np.issubdtype(a.dtype, np.floating)
    ]
    if floating_dtypes:
        # Return the floating dtype with the highest precision:
        return np.result_type(*floating_dtypes)

    # If none of the input arrays have a floating point dtype, they must be all
    # integer arrays or containers of Python scalars: return the default
    # floating point dtype for the namespace (implementation specific).
    return np.asarray(0.0).dtype


def _check_reg_targets_with_floating_dtype(
    y_true, y_pred, sample_weight, multioutput
):
    """Ensures y_true, y_pred, and sample_weight correspond to same
    regression task.

    Extends `_check_reg_targets` by automatically selecting a suitable
    floating-point data type for inputs using `_find_matching_floating_dtype`.

    Use this private method only when converting inputs to array
    API-compatibles.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,)

    multioutput : array-like or string in ['raw_values', 'uniform_average', \
        'variance_weighted'] or None
        None is accepted due to backward compatibility of r2_score().

    Returns
    -------
    type_true : one of {'continuous', 'continuous-multioutput'}
        The type of the true target data, as output by
        'utils.multiclass.type_of_target'.

    y_true : array-like of shape (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    multioutput : array-like of shape (n_outputs) or string in ['raw_values', \
        'uniform_average', 'variance_weighted'] or None
        Custom output weights if ``multioutput`` is array-like or
        just the corresponding argument if ``multioutput`` is a
        correct keyword.
    """
    dtype_name = _find_matching_floating_dtype(y_true, y_pred, sample_weight)

    y_type, y_true, y_pred, sample_weight, multioutput = _check_reg_targets(
        y_true, y_pred, sample_weight, multioutput, dtype=dtype_name
    )

    return y_type, y_true, y_pred, sample_weight, multioutput
