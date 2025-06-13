import numpy as np
from collections.abc import Mapping
from numpy import asanyarray
from torch.utils.data import Dataset
from sklearn.utils import validation
import torch
import tqdm

def to_numpy(X):
    """Generic function to convert a pytorch tensor to numpy.

    This function tries to unpack the tensor(s) from supported
    data structures (e.g., dicts, lists, etc.) but doesn't go
    beyond.

    Returns X when it already is a numpy array.

    """
    if isinstance(X, np.ndarray):
        return X

    if isinstance(X, Mapping):
        return {key: to_numpy(val) for key, val in X.items()}

    if isinstance(X, (tuple, list)):
        return type(X)(to_numpy(x) for x in X)

    if not isinstance(X, torch.Tensor):
        raise TypeError("Cannot convert this data type to a numpy array.")

    if X.is_cuda:
        X = X.cpu()

    if hasattr(X, 'is_mps') and X.is_mps:
        X = X.cpu()

    if X.requires_grad:
        X = X.detach()

    return X.numpy()


def get_labels_from_dataset(X, y=None, raise_if_none=False):
    """
    Parse the dataset X to collect the labels y and returns cat(y).
    It assumes that X[i] returns tuple (x, y)
    If `y` is given, only check if its size match with `X`.
    
    Parameters
    ----------
    X: torch.utils.data.dataset.Dataset or array
        Dataset to parse. If an array is given, it is not parsed.
    
    y: np.ndarray
        Target labels.
    
    raise_if_none: bool
        If True, raise ValueError if no labels available. Otherwise, returns None
    
    Returns
    ----------
    y: np.ndarray
    """
    if y is None and isinstance(X, Dataset):
        # Parse `X`
        y = []
        for elem in tqdm.tqdm(X, desc=f"Parsing of dataset {X.__class__.__name__}"):
            if isinstance(elem, tuple) and len(elem) == 2:
                if elem[1] is None:
                    break
                y.append(elem[1])
    if len(y) == 0:
        if raise_if_none:
            raise ValueError("No available labels")
        return None
    elif len(y) < len(X):
        raise ValueError("Target labels size %i must match dataset size %i"%(len(y), len(X)))
    else:
        y = atleast_2d(y)
        return y


def atleast_2d(X: np.ndarray):
    X = asanyarray(X)
    if X.ndim == 0:
        X = X.reshape(1, 1)
    elif X.ndim == 1:
        X = X[:, np.newaxis]
    return X

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


class AverageMeter:
    """Computes and stores the average, sum, and count of a metric."""
    
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all statistics."""
        self.val = 0  # Most recent value
        self.avg = 0  # Running average
        self.sum = 0  # Sum of all values
        self.count = 0  # Number of updates

    def update(self, val, n=1):
        """
        Updates the meter with a new value.
        
        Args:
            val (float): The new value to add.
            n (int): The number of occurrences (default: 1, useful for batch processing).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

    def __str__(self):
        """String representation of the current average."""
        return f"Avg: {self.avg:.4f}, Sum: {self.sum:.4f}, Count: {self.count}"