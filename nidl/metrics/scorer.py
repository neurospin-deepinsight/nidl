
class _BaseScorer():

    def __init__(self, scoring):
        if not isinstance(scoring, str):
            raise ValueError("scoring must be a string, got {}".format(type(scoring)))
        self._scoring = scoring

    def __call__(self, estimator, X, collate_fn):
        """Basic scorer that applies `score` method to estimator

        Parameters
        ----------
        estimator : object
            Estimator used to fit the data. 

        X : torch Dataset
            Test data that will be fed to `estimator.score`.

        collate_fn: BaseCollateFunction
            Collate function used to transform the data.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        if not hasattr(estimator, "score"):
            raise ValueError("estimator must have `score` method implemented")
        scores = estimator.score(X, collate_fn)
        if not isinstance(scores, dict):
            raise ValueError("scores must be dict, got {}".format(type(scores)))
        if self._scoring not in scores:
            raise ValueError("not score {} found in {}".format(self._scoring, scores))
        return scores[self._scoring]


def check_scoring(estimator, scoring):
    if callable(scoring):
        return scoring
    if not hasattr(estimator, "score"):
        raise ValueError("estimator must implement `score` method")
    scorer = _BaseScorer(scoring)
    return scorer
    


