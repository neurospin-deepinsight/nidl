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

def check_array(X):
    """
    Ensure the input is converted to a NumPy array, compatible with scikit-learn.
    Parameters
    ----------
    X: object
        Input object to check / convert.
    Returns:
        np.ndarray: The input converted to a NumPy array.
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()  # Convert Tensor to NumPy array
    
    # Validate and enforce NumPy format for scikit-learn compatibility
    return validation.check_array(X) 


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