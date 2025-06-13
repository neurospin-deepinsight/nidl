from inspect import isclass

def check_is_fitted(estimator, attributes=None, *, msg=None, all_or_any=all, check_is_none=False):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a :class:`~sklearn.exceptions.NotFittedError` with the given message.

    If an estimator does not set any attributes with a trailing underscore, it
    can define a ``__sklearn_is_fitted__`` method returning a boolean to
    specify if the estimator is fitted or not. See
    :ref:`sphx_glr_auto_examples_developing_estimators_sklearn_is_fitted.py`
    for an example on how to use the API.


    Parameters
    ----------
    estimator : estimator instance
        Estimator instance for which the check is performed.

    attributes : str, list or tuple of str, default=None
        Attribute name(s) given as string or a list/tuple of strings
        Eg.: ``["coef_", "estimator_", ...], "coef_"``

        If `None`, `estimator` is considered fitted if there exist an
        attribute that ends with a underscore and does not start with double
        underscore.

    msg : str, default=None
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        estimator."

        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.

        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default=all
        Specify whether all or any of the given attributes must exist.
    
    check_is_none: bool, default=False
        If True, checks if the attributes are None instead of checking their existence.
        This is useful for estimators that may have attributes set to None after fitting.

    Raises
    ------
    TypeError
        If the estimator is a class or not an estimator instance

    NotFittedError
        If the attributes are not found.

    """
    if isclass(estimator):
        raise TypeError("{} is a class, not an instance.".format(estimator))
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this estimator."
        )

    if not hasattr(estimator, "fit"):
        raise TypeError("%s is not an estimator instance." % (estimator))
    
    is_fitted = False 

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        is_fitted = all_or_any([hasattr(estimator, attr) and (not check_is_none or 
                                                              getattr(estimator, attr) is not None)for attr in attributes])
    else:
        fitted_attrs = [
            v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")
        ]
        is_fitted = len(fitted_attrs) > 0

    if not is_fitted:
        raise ValueError(msg % {"name": type(estimator).__name__})

