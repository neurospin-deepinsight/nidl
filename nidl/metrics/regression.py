##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from ._regression import _check_reg_targets_with_floating_dtype

"""Sets of functions to compute regression metrics to evaluate the
performance of a regressor."""


def pearson_r(
    y_true,
    y_pred,
    sample_weight=None,
    multioutput="uniform_average",
    force_finite=False,
):
    """Pearson correlation coefficient between 2 arrays y_true, y_pred.
    This score is symmetric between y_true and y_pred and is always between 1
    (perfect correlation) and -1 (perfect anti-correlation).

    Parameters
    ----------
    y_true: array-like of shape (n_samples,) or (n_samples, n_outputs)
        First input array.

    y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Second input array.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights for weighted Pearson's correlation.

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
        (n_outputs,), optional
        Defines aggregating of multiple output scores.

        - 'raw_values': Returns a full set of scores in case of multioutput
          input.
        - 'uniform_average': Scores of all outputs are averaged with uniform
          weight.
        - array-like: Defines weights used to average scores.
    
    force_finite : bool, default=False
        Flag indicating if ``NaN`` and ``-Inf`` scores resulting from constant
        data should be replaced with real numbers (``1.0`` if prediction is
        perfect, i.e. all data are equal , ``0.0`` otherwise).
        Default is ``False`` since Pearson's correlation is not defined
        for constant data (zero variance).


    Returns
    -------
    pearson_r: float or array of floats
        The correlation score or ndarray of scores if 'multioutput' is
        'raw_values'.

    """
    _, y_true, y_pred, sample_weight, multioutput = (
        _check_reg_targets_with_floating_dtype(
            y_true, y_pred, sample_weight, multioutput
        )
    )
    # If weights are None or all zeros, assume uniform weights
    if sample_weight is None:
        sample_weight = np.ones((len(y_true), 1), dtype=y_true.dtype)

    if sample_weight.ndim == 1:
        sample_weight = sample_weight[:, np.newaxis]

    w_sum = np.sum(sample_weight)

    x_mean = np.sum(sample_weight * y_true, axis=0) / w_sum
    y_mean = np.sum(sample_weight * y_pred, axis=0) / w_sum

    # Division by w_sum is omitted as it is unnecessary
    cov_xy = np.sum(
        sample_weight * (y_true - x_mean) * (y_pred - y_mean), axis=0
    )
    var_x = np.sum(sample_weight * (y_true - x_mean) ** 2, axis=0)
    var_y = np.sum(sample_weight * (y_pred - y_mean) ** 2, axis=0)

    denom = np.sqrt(var_x * var_y)

    if not force_finite:
        # If denom is zero, result is nan or inf
        output_scores = cov_xy / denom
    else:
        n_outputs = cov_xy.shape[0]
        nonzero_denominator = denom != 0
        if y_pred.shape[0] > 0:
            equal_data = (
                (var_x == 0)
                & (var_y == 0)
                & np.allclose(y_pred[0, :], y_true[0, :])
            )
        else:
            equal_data = np.zeros(n_outputs, dtype=bool)
        output_scores = np.ones(n_outputs, dtype=cov_xy.dtype)
        # Non-zero Denominator: use the formula
        valid_score = nonzero_denominator

        output_scores[valid_score] = cov_xy[valid_score] / denom[valid_score]

        # Data are not equal and Zero Denominator:
        # arbitrary set to 0.0 to avoid -inf scores
        output_scores[~equal_data & ~nonzero_denominator] = 0.0

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            # return scores individually
            return output_scores
        elif multioutput == "uniform_average":
            # passing None as weights results is uniform mean
            avg_weights = None
        else:
            raise ValueError(f"multioutput {multioutput} not implemented")
    else:
        avg_weights = None

    return np.average(output_scores, weights=avg_weights)


def regression_report(
    y_true,
    y_pred,
    *,
    target_names=None,
    sample_weight=None,
    digits=2,
    output_dict=False,
):
    """Build a text report showing the main regression metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated targets as returned by a regressor.

    target_names : array-like of shape (n_outputs,), default=None
        Optional display names by regressor (following index order).

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    digits : int, default=2
        Number of digits for formatting output floating point values.
        When ``output_dict`` is ``True``, this will be ignored and the
        returned values will not be rounded.

    output_dict : bool, default=False
        If True, return output as dict.

    Returns
    -------
    report : str or dict
        Text summary of the mean absolute error, median absolute error, root
        mean squared error, mean squared error, r2 score, pearsonr, and
        explained variance score for each regressor. The uniform average
        across all regressors is also computed and reported.
        Dictionary returned if output_dict is True. Dictionary has the
        following structure::

            { 'regressor 0': {
                'MAE': 0.5,     # Mean Absolute Error
                'MedAE': 0.5,   # Median Absolute Error
                'RMSE': 0.6,    # Root Mean Squared Error
                'MSE': 0.4,     # Mean Squared Error
                'R2': 0.8,      # R2 score
                'PCC': 0.9,     # Pearson Correlation Coefficient
                'EVar': 0.7,    # Explained Variance score
                },
              'regressor 1': { ... },
               ...
              'average': {
                 'MAE': 0.6,
                 'MedAE': 0.6,
                 'RMSE': 0.7,
                 'MSE': 0.5,
                 'R2': 0.75,
                 'PCC': 0.85,
                 'EVar': 0.65
                 }
            }

        The reported average is computed as the arithmetic mean
        across all regressors. If only one regressor is available, the
        metrics are directly reported (without specifying 'regressor 0' or the
        'average').

    Examples
    --------
    >>> from nidl.metrics import regression_report
    >>> y_true = [[0.1, 1.2], [5.1, 2.0], [2.0, 3.0], [4.0, 2.0]]
    >>> y_pred = [[0.0, 0.0], [2.0, 2.0], [1.0, 1.0], [1.0, 1.0]]
    >>> print(regression_report(y_true, y_pred))
                    MAE   MedAE   RMSE    MSE     R2   PCC  EVar

    regressor 0     1.80   2.00   2.21   4.90  -0.34  0.92  0.55
    regressor 1     1.05   1.10   1.27   1.61  -2.95  0.44 -0.25

    average         1.42   1.55   1.74   3.26  -1.64  0.68  0.15

    >>> y_true = [0.1, 5.1, 2.0, 4.0]
    >>> y_pred = [0.0, 2.0, 1.0, 1.0]
    >>> print(regression_report(y_true, y_pred))
                    MAE   MedAE   RMSE    MSE     R2   PCC  EVar
    regressor 0    1.80   2.00   2.21   4.90  -0.34  0.92  0.55

    """
    headers = [
        "MAE",
        "MedAE",
        "RMSE",
        "MSE",
        "R2",
        "PCC",
        "EVar",
    ]
    # compute per-regressor results without averaging
    metrics = [
        mean_absolute_error,
        median_absolute_error,
        root_mean_squared_error,
        mean_squared_error,
        r2_score,
        pearson_r,
        explained_variance_score,
    ]
    metrics_computed = [
        metric(
            y_true,
            y_pred,
            multioutput="raw_values",
            sample_weight=sample_weight,
        )
        for metric in metrics
    ]
    n_regressors = len(metrics_computed[0])
    if target_names is None:
        target_names = [f"regressor {i}" for i in range(n_regressors)]
    elif len(target_names) != len(metrics_computed[0]):
        raise ValueError(
            f"Number of regressors {n_regressors} does not match size of "
            f"target_name {len(target_names)} "
        )

    rows = zip(target_names, *metrics_computed)
    last_line_heading = "average"
    if output_dict:
        report_dict = {label[0]: label[1:] for label in rows}
        for label, scores in report_dict.items():
            report_dict[label] = dict(zip(headers, [float(i) for i in scores]))
    else:
        name_width = max(len(cn) for cn in headers)
        num_metrics = len(headers)
        width = max(name_width, len("regressor 0"), digits)
        head_fmt = "{:>{width}s} " + " {:>8}" * num_metrics
        report = head_fmt.format("", *headers, width=width)
        report += "\n\n"
        row_fmt = "{:>{width}s} " + " {:>8.{digits}f}" * num_metrics + "\n"
        for row in rows:
            report += row_fmt.format(*row, width=width, digits=digits)
        report += "\n"

    # compute average metrics across all regressors
    avg = [np.mean(metric, axis=0) for metric in metrics_computed]

    if output_dict:
        if n_regressors > 1:
            report_dict[last_line_heading] = dict(
                zip(headers, [float(i) for i in avg])
            )
        elif n_regressors == 1:
            return report_dict[target_names[0]]
        return report_dict
    else:
        if n_regressors > 1:
            report += row_fmt.format(
                last_line_heading, *avg, width=width, digits=digits
            )
        return report
