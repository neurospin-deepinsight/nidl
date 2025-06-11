import torch
from scipy import sparse
from typing import Callable, Optional
from scipy import sparse

class Dataset(torch.utils.data.Dataset):
    r"""General dataset wrapper that can be used in conjunction with
    PyTorch :class:`~torch.utils.data.DataLoader`.

    The dataset will always yield a tuple of two values, the first
    from the data (``X``) and the second from the target (``y``).
    However, the target is allowed to be ``None``. 

    :class:`.Dataset` currently works with the following data types:

    * numpy ``array``\s
    * PyTorch :class:`~torch.Tensor`\s
    * scipy sparse CSR matrices

    Parameters
    ----------
    X : see above
      Everything pertaining to the input data.

    y : see above or None, default=None
      Everything pertaining to the target, if there is anything.

    transform: callable or None, default=None
        Transformation to apply to the input data X[i].
    
    target_transform: callable or None, default=None
        Transformation to apply to the target data y[i] (if given).
    """
    def __init__(
            self,
            X,
            y=None,
            transform: Optional[Callable]=None,
            target_transform: Optional[Callable]=None):
        
        self.X = X
        self.y = y
        self.transform = transform
        self.target_transform = target_transform

        # Convert it to dense for now
        if self._issparse(X):
            self.X = X.toarray().squeeze(0)

        len_X = len(X)
        if y is not None:
            len_y = len(y)
            if len_y != len_X:
                raise ValueError("X and y have inconsistent lengths.")
        self._len = len_X

    def _is_sparse(self, x):
        try:
            return sparse.issparse(x) or x.is_sparse
        except AttributeError:
            return False

    def __len__(self):
        return self._len


    def __getitem__(self, i):
        Xi = self.X[i]
        yi = self.y[i] if self.y is not None else None

        if self.transform is not None:
            Xi = self.transform(Xi)
        
        if self.target_transform is not None and yi is not None:
            yi = self.target_transform(yi)
        
        return Xi, yi

