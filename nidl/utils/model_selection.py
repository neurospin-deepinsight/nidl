import numpy as np
import sklearn
import sklearn.model_selection
from sklearn.utils.validation import validate_data
from skmultilearn.model_selection.iterative_stratification import IterativeStratification


class StratifiedShuffleSplit(sklearn.model_selection.StratifiedShuffleSplit):
    """Iteratively stratifies train/test split and deals both with discrete 
    and continuous labels for stratification.

    Warning: for now, splits across folds are equal. 

    This class requires scikit-multilearn library that can be installed with
    $ pip install scikit-multilearn
    More information on https://github.com/scikit-multilearn/scikit-multilearn

    Parameters
    ----------
    n_splits : int, default=10
        Number of re-shuffling & splitting iterations.
    
    test_size : float, [0,1], default=0.1
        The proportion of the dataset to include in the test split, the rest will be put in the train set

    discretize: bool, default=False
        If True, discretize the label data using KBinsDiscretizer strategy which automatically bins the 
        label using `numpy.histogram_bin_edges` method. Set True if you deal with continous labels.

    random_state: int, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
    """
    def __init__(self, 
                 n_splits: int=10, 
                 test_size: float=0.1, 
                 discretize: bool=False, 
                 random_state: int=None):
        
        super().__init__(n_splits=n_splits, test_size=test_size, random_state=random_state)

        self.discretize = discretize
        self.splitter = IterativeStratification(
            n_splits=2, order=2, 
            sample_distribution_per_fold=[self.test_size, 1.0-self.test_size]
        )

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.

        y : array-like of shape (n_samples,) or (n_samples, n_labels)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        if self.discretize:
            y = KBinsDiscretizer(n_bins='auto', encode='ordinal').fit_transform(y)
        
        for i in range(self.n_splits):
            yield next(iter(self.splitter.split(X, y)))


class KBinsDiscretizer(sklearn.preprocessing.KBinsDiscretizer):

    """
    Bin continuous data into intervals.

    Additional features compared to original sklearn method: it
    can compute automatically the optimal bins number for each
    feature according to 'auto' strategy from
    `numpy.histogram_bin_edges` method. It also accepts several
    estimation methods from the literature.

    Parameters
    ----------
    n_bins : str, int, array-like of shape (n_features,), default='auto'
        The number of bins to produce. Raises ValueError if ``n_bins < 2``.
        If str, must be in {'auto', 'fd' (Freedman Diaconis), 'doane' (Improved Sturges),
                            'scott', 'rice', 'sturges', 'sqrt', 'stone'}
        It estimates the optimal number of bins for each feature during call to `fit`.
        Refer to https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html
        for more details on these estimators.

    **kwargs:
        Other keyword arguments given to super() constructor. Please refer to
        sklearn.preprocessing.KBinsDiscretizer

    Attributes
    ----------
    bin_edges_ : ndarray of ndarray of shape (n_features,)
        The edges of each bin. Contain arrays of varying shapes ``(n_bins_, )``
        Ignored features will have empty arrays.

    n_bins_ : ndarray of shape (n_features,), dtype=np.int_
        Number of bins per feature. Bins whose width are too small
        (i.e., <= 1e-8) are removed with a warning.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    """
    def __init__(self, n_bins='auto', **kwargs):
        super().__init__(n_bins=n_bins, **kwargs)
    
    def fit(self, X, y=None):
        """
          Fit the estimator.

          Parameters
          ----------
          X : array-like of shape (n_samples, n_features)
              Data to be discretized.

          y : None
              Ignored. This parameter exists only for compatibility with
              :class:`~sklearn.pipeline.Pipeline`.

          Returns
          -------
          self : object
              Returns the instance itself.
        """
        X = validate_data(self, X, dtype="numeric")
        if isinstance(self.n_bins, str):
            supported_estimators = {'auto', 'fd', 'doane', 'scott',
                                   'rice', 'sturges', 'sqrt', 'stone'}
            if self.n_bins not in supported_estimators:
                raise ValueError("n_bins must be in {}".format(supported_estimators))
            optimal_n_bins = []
            for i in range(X.shape[1]):
                bin_edges = np.histogram_bin_edges(X[:, i], bins=self.n_bins)
                optimal_n_bins.append(len(bin_edges)-1)
            _original_n_bins = self.n_bins
            self.n_bins = optimal_n_bins
            super().fit(X, y)
            self.n_bins = _original_n_bins
        else:
            super().fit(X, y)
        return self
