from sklearn.model_selection import GridSearchCV
from sklearn.model_selection._split import check_cv
from sklearn.utils.validation import indexable
from sklearn.utils.parallel import Parallel, delayed
from sklearn.base import clone, is_classifier
from torch.utils.data.dataset import Subset
from torch.utils.data import Dataset
from contextlib import suppress
import numpy as np
import time
import numbers
import warnings
from joblib import logger
from traceback import format_exc
from itertools import product
from collections import defaultdict
from nidl.metrics.scorer import check_scoring


class GridSearch(GridSearchCV):
    """
    Exhaustive search over specified parameter values for an estimator using scikit-learning API
    and Torch API for input data format, allowing large-scale training. Parameters are validated
    using cross-validation and the final estimator is refitted on training set using the best params
    found.

    Warning: the scoring used must follow the scikit-learn convention: the higher the better. It is NOT
    checked by this estimator.
    """

    def fit(self, X: Dataset, y=None, *,  groups=None, **fit_params):
        """Run fit with all sets of parameters.

            Parameters
            ----------

            X : torch.Dataset of shape (n_samples, *)
                Training vectors (with eventually labels), where `n_samples`
                is the number of samples. It must implement __len__() and __getitem__()
                methods and it also contains target values

            y: array-like of shape (n_samples, n_output) or (n_samples,), default=None
                Target relative to X for classification or regression;
                None for unsupervised learning.

            groups : array-like of shape (n_samples,), default=None
                Group labels for the samples used while splitting the dataset into
                train/test set. Only used in conjunction with a "Group" :term:`cv`
                instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).

            **fit_params : dict of str -> object
                Parameters passed to the `fit` method of the estimator.

                If a fit parameter is an array-like whose length is equal to
                `num_samples` then it will be split across CV groups along with `X`
                and `y`. For example, the :term:`sample_weight` parameter is split
                because `len(sample_weights) = len(X)`.

            Returns
            -------
            self : object
                Instance of fitted estimator.
        """
        estimator = self.estimator
        refit_metric = "score"

        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(estimator, self.scoring)
        elif isinstance(self.scoring, (list, set, tuple, dict)):
            raise NotImplementedError("scoring must be a string or callable function "
                                      "(no multimetric scoring implemented)")
        else:
           raise ValueError("scoring not implemented: {}".format(self.scoring))

        if not isinstance(X, Dataset):
            raise ValueError("Input data must be a torch dataset (got {})".format(type(X)))

        if not hasattr(X, "__getitem__") or not hasattr(X, "__len__"):
            raise ValueError("Dataset {} must implement __getitem__() and __len__()".format(X))

        X, y, groups = indexable(X, y, groups)

        # Creates dummy np.ndarray X with shape (n_samples, 1) for compatibility with sklearn API
        X_dummy = np.zeros((len(X), 1), dtype=np.float16)

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X_dummy, y, groups)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=fit_params,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )
        results = {}

        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None, more_results=None):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        "Fitting {0} folds for each of {1} candidates,"
                        " totalling {2} fits".format(
                            n_splits, n_candidates, n_candidates * n_splits
                        )
                    )

                out = parallel(
                    delayed( GridSearch._fit_and_score)(
                        clone(base_estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        parameters=parameters,
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(cand_idx, n_candidates),
                        **fit_and_score_kwargs)
                    for (cand_idx, parameters), (split_idx, (train, test)) in product(
                        enumerate(candidate_params), enumerate(cv.split(X_dummy, y, groups))
                    )
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. "
                        "Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned "
                        "inconsistent results. Expected {} "
                        "splits, got {}".format(n_splits, len(out) // n_candidates)
                    )

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results
                )

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]["test_scores"]
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = self._select_best_index(
                self.refit, refit_metric, results
            )
            if not callable(self.refit):
                # With a non-custom callable, we can select the best score
                # based on the best index
                self.best_score_ = results[f"mean_test_{refit_metric}"][
                    self.best_index_
                ]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(
                clone(base_estimator).set_params(**self.best_params_)
            )
            refit_start_time = time.time()
            self.best_estimator_.fit(X, y=y, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

    @staticmethod
    def _fit_and_score(
            estimator,
            X,
            y,
            scorer,
            train,
            test,
            verbose,
            parameters,
            fit_params,
            return_train_score=False,
            return_parameters=False,
            return_n_test_samples=False,
            return_times=False,
            return_estimator=False,
            split_progress=None,
            candidate_progress=None,
            error_score=np.nan,
    ):

        """Fit estimator and compute scores for a given dataset split.

        Parameters
        ----------
        estimator : estimator object implementing 'fit'
            The object to use to fit the data.

        X : torch.Dataset with shape (n_samples, *)
            The data to fit.

        y:  array-like of shape (n_samples,) or (n_samples, n_outputs) or None
            The target variable to try to predict in the case of
            supervised learning.

        scorer : A single callable
            The return value for ``train_scores`` and
            ``test_scores`` is a single float.

            The callable object / fn should have signature
            ``scorer(estimator, X, collate_fn)`` where X is a torch Dataset.

        train : array-like of shape (n_train_samples,)
            Indices of training samples.

        test : array-like of shape (n_test_samples,)
            Indices of test samples.

        verbose : int
            The verbosity level.

        error_score : 'raise' or numeric, default=np.nan
            Value to assign to the score if an error occurs in estimator fitting.
            If set to 'raise', the error is raised.
            If a numeric value is given, FitFailedWarning is raised.

        parameters : dict or None
            Parameters to be set on the estimator.

        fit_params : dict or None
            Parameters that will be passed to ``estimator.fit``.

        return_train_score : bool, default=False
            Compute and return score on training set.

        return_parameters : bool, default=False
            Return parameters that has been used for the estimator.

        split_progress : {list, tuple} of int, default=None
            A list or tuple of format (<current_split_id>, <total_num_of_splits>).

        candidate_progress : {list, tuple} of int, default=None
            A list or tuple of format
            (<current_candidate_id>, <total_number_of_candidates>).

        return_n_test_samples : bool, default=False
            Whether to return the ``n_test_samples``.

        return_times : bool, default=False
            Whether to return the fit/score times.

        return_estimator : bool, default=False
            Whether to return the fitted estimator.

        Returns
        -------
        result : dict with the following attributes
            train_scores : dict of scorer name -> float
                Score on training set (for all the scorers),
                returned only if `return_train_score` is `True`.
            test_scores : dict of scorer name -> float
                Score on testing set (for all the scorers).
            n_test_samples : int
                Number of test samples.
            fit_time : float
                Time spent for fitting in seconds.
            score_time : float
                Time spent for scoring in seconds.
            parameters : dict or None
                The parameters that have been evaluated.
            estimator : estimator object
                The fitted estimator.
            fit_error : str or None
                Traceback str if the fit failed, None if the fit succeeded.
        """
        if not isinstance(error_score, numbers.Number) and error_score != "raise":
            raise ValueError(
                "error_score must be the string 'raise' or a numeric value. "
                "(Hint: if using 'raise', please make sure that it has been "
                "spelled correctly.)"
            )

        progress_msg = ""
        if verbose > 2:
            if split_progress is not None:
                progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
            if candidate_progress and verbose > 9:
                progress_msg += f"; {candidate_progress[0]+1}/{candidate_progress[1]}"

        if verbose > 1:
            if parameters is None:
                params_msg = ""
            else:
                sorted_keys = sorted(parameters)  # Ensure deterministic o/p
                params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
        if verbose > 9:
            start_msg = f"[CV{progress_msg}] START {params_msg}"
            print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

        # Creates dummy np.ndarray X with shape (n_samples, 1) for compatibility with sklearn API
        X_dummy = np.zeros((len(X), 1), dtype=np.float16)

        # Adjust length of sample weights
        fit_params = fit_params if fit_params is not None else {}

        if parameters is not None:
            # clone after setting parameters in case any parameters
            # are estimators (like pipeline steps)
            # because pipeline doesn't clone steps in fit
            cloned_parameters = {}
            for k, v in parameters.items():
                cloned_parameters[k] = clone(v, safe=False)

            estimator = estimator.set_params(**cloned_parameters)

        start_time = time.time()

        X_train = Subset(X, indices=train)
        if y is not None:
            y_train = y[train]
        else:
            y_train = None
        X_test = Subset(X, indices=test)

        result = {}
        try:
            estimator.fit(X_train, y=y_train, **fit_params)
        except Exception:
            # Note fit time as time until error
            fit_time = time.time() - start_time
            score_time = 0.0
            if error_score == "raise":
                raise
            elif isinstance(error_score, numbers.Number):
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
            result["fit_error"] = format_exc()
        else:
            result["fit_error"] = None

            fit_time = time.time() - start_time
            test_scores = GridSearch._score(estimator, X_test, scorer, error_score)
            score_time = time.time() - start_time - fit_time
            if return_train_score:
                train_scores = GridSearch._score(estimator, X_train, scorer, error_score)

        if verbose > 1:
            total_time = score_time + fit_time
            end_msg = f"[CV{progress_msg}] END "
            result_msg = params_msg + (";" if params_msg else "")
            if verbose > 2:
                if isinstance(test_scores, dict):
                    for scorer_name in sorted(test_scores):
                        result_msg += f" {scorer_name}: ("
                        if return_train_score:
                            scorer_scores = train_scores[scorer_name]
                            result_msg += f"train={scorer_scores:.3f}, "
                        result_msg += f"test={test_scores[scorer_name]:.3f})"
                else:
                    result_msg += ", score="
                    if return_train_score:
                        result_msg += f"(train={train_scores:.3f}, test={test_scores:.3f})"
                    else:
                        result_msg += f"{test_scores:.3f}"
            result_msg += f" total time={logger.short_format_time(total_time)}"

            # Right align the result_msg
            end_msg += "." * (80 - len(end_msg) - len(result_msg))
            end_msg += result_msg
            print(end_msg)

        result["test_scores"] = test_scores
        if return_train_score:
            result["train_scores"] = train_scores
        if return_n_test_samples:
            result["n_test_samples"] = len(X_test)
        if return_times:
            result["fit_time"] = fit_time
            result["score_time"] = score_time
        if return_parameters:
            result["parameters"] = parameters
        if return_estimator:
            result["estimator"] = estimator
        return result

    @staticmethod
    def _score(estimator, X_test, scorer, error_score="raise"):
        """Compute the score(s) of an estimator on a given test set.

        Will return a dict of floats if `scorer` is a dict, otherwise a single
        float is returned.
        """

        try:
            scores = scorer(estimator, X_test)
        except Exception:
            if error_score == "raise":
                raise
            else:
                scores = error_score
                warnings.warn(
                    "Scoring failed. The score on this train-test partition for "
                    f"these parameters will be set to {error_score}. Details: \n"
                    f"{format_exc()}",
                    UserWarning,
                )

        error_msg = "scoring must return a number, got %s (%s) instead. (scorer=%s)"
        if hasattr(scores, "item"):
            with suppress(ValueError):
                # e.g. unwrap memmapped scalars
                scores = scores.item()
        if not isinstance(scores, numbers.Number):
            raise ValueError(error_msg % (scores, type(scores), scorer))
        return scores

