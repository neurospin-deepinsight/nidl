from warnings import simplefilter
from torch.utils.data import DataLoader
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, mean_absolute_error, \
    root_mean_squared_error, r2_score
from scipy.stats import pearsonr
from typing import List, Union
from pytorch_lightning import Callback
# Local import
from nidl.utils.util import check_array

# Silence repeated convergence warnings from scikit-learn logistic regression.
simplefilter("ignore", category=ConvergenceWarning)


class RidgeCVCallback(Callback):
    """Perform Ridge regression on top of a `torch.nn.Module` model trained by a 
    `neuroclav.solvers.base.Solver`. 
      
      Concretely, after each validation loop or each fit, this callback:
        1) Embeds the input data through the torch model
        2) Performs n-fold CV to find the best l2 regularization strength
        3) Log the main regression metrics including:
            * root mean squared error
            * R^2: coefficient of determination
            * mean absolute error
            * Pearson-r: correlation coefficient
    """

    def __init__(
               self,
               train_dataloader: DataLoader,
               test_dataloader: DataLoader,
               dataset_name: Union[str, None]=None,
               alphas: List[float]=[0.1, 1.0, 10.0],
               cv: int=5,
               scoring: str="r2",
               frequency: str="by_epoch",
               **extraction_kwargs):
        """
        Parameters
        ----------
        train_dataloader: torch.utils.data.DataLoader
            Training dataloader to use for fitting the classifier. 

        test_dataloader: torch.utils.data.DataLoader
            Testing dataloader to use for evaluating the classifier. 
        
        dataset_name: str or None
            Name of the dataset to be used for classification. This name
            will appear when logging the results as `ridge_probe_<dataset_name>`.
            If None, only `ridge_probe` is displayed.
        
        alphas : list of floats, default=[0.1, 1.0, 10.0]
            Arrays of `alpha` values to try in CV. It corresponds to the regularization strenght.

        cv: int or cross-validation generator, default=5
            How many folds to use for cross-validating the `alpha` regularization strenght 
            in the `Ridge` regression. 
        
        scoring: str in {"r2", "neg_mean_absolute_error", "neg_mean_squared_error", ...}, 
            default="r2" 
            Which scoring function to use to cross-validate the `alpha` hyper-parameter.
            For a complete list of scoring options, check 
            https://scikit-learn.org/1.4/modules/model_evaluation.html#scoring

        frequency: str in {'by_epoch', 'by_fit'}
            When to apply the linear probing. Either 'by_epoch', after each validation loop 
            or 'by_fit', only once at the end of training.

        extraction_kwargs: dict
            Keyword arguments given to `solver.get_embedding` method
        """

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.dataset_name = dataset_name
        self.alphas = alphas
        self.cv = cv
        self.scoring = scoring
        self.frequency = frequency
        self.extraction_kwargs = extraction_kwargs

    def linear_probing(self, solver, log: bool=True):
            
        if not hasattr(solver, "get_embedding"):
            raise ValueError("`get_embedding` must be implemented for linear probing")
        
        # Embed the data
        X_train, y_train = solver.get_embedding(self.train_dataloader, **self.extraction_kwargs)
        X_test, y_test = solver.get_embedding(self.test_dataloader, **self.extraction_kwargs)

        # Check arrays
        X_train, y_train = check_array(X_train), check_array(y_train)
        X_test, y_test = check_array(X_test), check_array(y_test)

        # Fit the regressor
        linear_probe = RidgeCV(
            alphas=self.alphas,
            cv=self.cv,
            scoring=self.scoring,
        )
        linear_probe.fit(X_train, y_train)

        # Make predictions
        predictions = linear_probe.predict(X_test)

        # Compute regression metrics
        metrics_report = dict()
        metrics_report["MAE"] = mean_absolute_error(y_test, predictions)
        metrics_report["RMSE"] = root_mean_squared_error(y_test, predictions)
        metrics_report["R2"] = r2_score(y_test, predictions)
        metrics_report["pearson_r"] = pearsonr(y_test.flatten(), predictions.flatten()).statistic

        # Log the results
        if log:
            base_name = f"ridge_probe_{self.dataset_name}" \
                if self.dataset_name is not None else "ridge_probe"
            for name, value in metrics_report.items():
                solver.log(f"{base_name}/{name}", value)
            
        # Return all metrics
        return metrics_report

    def on_validation_epoch_end(self, solver):
        if self.frequency == "by_epoch":
            self.linear_probing(solver)

    def on_fit_end(self, solver):
        if self.frequency == "by_fit":
            self.linear_probing(solver)


class KNeighborsRegressorCVCallback(Callback):
    """Perform KNN regression on top of a `torch.nn.Module` model trained by a 
    `neuroclav.solvers.base.Solver`. 
      
      Concretely, after each validation loop or each fit, this callback:
        1) Embeds the input data through the torch model
        2) Performs n-fold CV to find the best `n_neighbors` neighbors
        3) Log the main regression metrics including:
            * root mean squared error
            * R^2: coefficient of determination
            * mean absolute error
            * Pearson-r: correlation coefficient
    """

    def __init__(
               self,
               train_dataloader: DataLoader,
               test_dataloader: DataLoader,
               dataset_name: Union[str, None]=None,
               n_neighbors: List[int]=[2, 5, 10],
               cv: int=5,
               n_jobs: Union[int, None]=None,
               scoring: str="r2",
               frequency: str = "by_epoch",
               **extraction_kwargs):
        """
        Parameters
        ----------
        train_dataloader: torch.utils.data.DataLoader
            Training dataloader to use for fitting the classifier. 

        test_dataloader: torch.utils.data.DataLoader
            Testing dataloader to use for evaluating the classifier. 
        
        dataset_name: str
            Name of the dataset to be used for classification. This name
            will appear when logging the results as `knn_regression_probe_<dataset_name>`.
            If None, only `knn_regression_probe` is displayed.

        n_neighbors: list of int, default=[2, 5, 10]
            Arrays of `n_neighbors` values to try in CV. 
            It corresponds to the number of neighbors to use by the KNN on the training set.

        cv: int or cross-validation generator, default=5
            How many folds to use for cross-validating the `alpha` regularization strenght 
            in the `Ridge` regression. 
        
        n_jobs: int or None, default=None
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. 

        scoring: str in {"r2", "neg_mean_absolute_error", "neg_mean_squared_error", ...}, 
            default="r2" 
            Which scoring function to use to cross-validate the `n_neighbors` hyper-parameter.
            For a complete list of scoring options, check 
            https://scikit-learn.org/1.4/modules/model_evaluation.html#scoring

        frequency: str in {'by_epoch', 'by_fit'}
            When to apply the linear probing. Either 'by_epoch', after each validation loop 
            or 'by_fit', only once at the end of training.

        extraction_kwargs: dict
            Keyword arguments given to `solver.get_embedding` method
        """

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.dataset_name = dataset_name
        self.n_neighbors = n_neighbors
        self.cv = cv
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.frequency = frequency
        self.extraction_kwargs = extraction_kwargs
    
    def knn_probing(self, solver, log: bool=True):
            
        if not hasattr(solver, "get_embedding"):
            raise ValueError("`get_embedding` must be implemented for linear probing")
        
        # Embed the data
        X_train, y_train = solver.get_embedding(self.train_dataloader, **self.extraction_kwargs)
        X_test, y_test = solver.get_embedding(self.test_dataloader, **self.extraction_kwargs)

        # Check arrays
        X_train, y_train = check_array(X_train), check_array(y_train)
        X_test, y_test = check_array(X_test), check_array(y_test)

        # Fit the regressor
        linear_probe = GridSearchCV(
            KNeighborsRegressor(),
            param_grid={"n_neighbors": self.n_neighbors},
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            cv=self.cv)
        linear_probe.fit(X_train, y_train)

        # Make predictions
        predictions = linear_probe.predict(X_test)

        # Check arrays
        X_train, y_train = check_array(X_train), check_array(y_train)
        X_test, y_test = check_array(X_test), check_array(y_test)

        # Compute regression metrics
        metrics_report = dict()
        metrics_report["MAE"] = mean_absolute_error(y_test, predictions)
        metrics_report["RMSE"] = root_mean_squared_error(y_test, predictions)
        metrics_report["R2"] = r2_score(y_test, predictions)
        metrics_report["pearson_r"] = pearsonr(y_test.flatten(), predictions.flatten()).statistic

        # Log the results
        if log:
            base_name = f"knn_regression_probe_{self.dataset_name}" \
                if self.dataset_name is not None else "knn_regression_probe"
            for name, value in metrics_report.items():
                solver.log(f"{base_name}/{name}", value)
            
        # Return all metrics
        return metrics_report

    def on_validation_epoch_end(self, solver):
        if self.frequency == "by_epoch":
            self.knn_probing(solver)

    def on_fit_end(self, solver):
        if self.frequency == "by_fit":
            self.knn_probing(solver)


class LogisticRegressionCVCallback(Callback):
    """Performs logistic regression (classification) on top of a `torch.nn.Module` 
    trained by a `neuroclav.solvers.base.Solver`.

    Concretely, after each validation loop or each fit, this callback:
        1) Embeds the input data through the torch model
        2) Performs n-fold CV to find the best l2 regularization strength
        3) Log the main classification metrics including:
            * precision, recall, f1-score, support for each class + averaged overall 
            * global accuracy
            * ROC-AUC 
    """
    
    def __init__(self, 
                 train_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 dataset_name: Union[str, None]=None,
                 Cs: Union[int, List[float]]=5,
                 cv: int=5,
                 max_iter: int=100,
                 n_jobs: int=None,
                 scoring: str="balanced_accuracy",
                 linear_solver: str="lbfgs",
                 frequency: str = "by_epoch",
                 **extraction_kwargs):
        """
        Parameters
        ----------
        train_dataloader: torch.utils.data.DataLoader
            Training dataloader to use for fitting the classifier. 

        test_dataloader: torch.utils.data.DataLoader
            Testing dataloader to use for evaluating the classifier. 
        
        dataset_name: str
            Name of the dataset to be used for classification. This name
            will appear when logging the results as `logistic_probe_<dataset_name>`.
            If None, only `logistic_probe` is displayed.
        
        Cs : int or list of floats, default=10
            Each of the values in Cs describes the inverse of regularization
            strength. If Cs is as an int, then a grid of Cs values are chosen
            in a logarithmic scale between 1e-4 and 1e4.
            Like in support vector machines, smaller values specify stronger
            regularization.

        cv: int or cross-validation generator, default=5
            How many folds to use for cross-validating the `C` regularization strenght 
            in the `LogisticRegression`. 
        
        max_iter: int, default=100
            Maximum number of iterations taken for the solver to converge.

        n_jobs: int, default=None
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. 
        
        scoring: str in {"accuracy", "balanced_accuracy", "f1", ...}, default="balanced_accuracy" 
            Which scoring function to use to cross-validate the `C` hyper-parameter.
            For a complete list of scoring options, check 
            https://scikit-learn.org/1.4/modules/model_evaluation.html#scoring
        
        linear_solver: str in {'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'}, \
            default='lbfgs'
            Algorithm to use in the optimization problem.

        frequency: str in {'by_epoch', 'by_fit'}
            When to apply the linear probing. Either 'by_epoch', after each validation loop 
            or 'by_fit', only once at the end of training.

        extraction_kwargs: dict
            Keyword arguments given to `solver.get_embedding` method
        """
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.dataset_name = dataset_name
        self.Cs = Cs
        self.cv = cv
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.linear_solver = linear_solver
        self.frequency = frequency
        self.extraction_kwargs = extraction_kwargs

    def linear_probing(self, solver, log: bool=True):
            
            if not hasattr(solver, "get_embedding"):
                raise ValueError("`get_embedding` must be implemented for linear probing")
            # Embed the data
            X_train, y_train = solver.get_embedding(self.train_dataloader, **self.extraction_kwargs)
            X_test, y_test = solver.get_embedding(self.test_dataloader, **self.extraction_kwargs)

            # Check arrays
            X_train, y_train = check_array(X_train), check_array(y_train)
            X_test, y_test = check_array(X_test), check_array(y_test)

            # Fit the classifier
            linear_probe = LogisticRegressionCV(
                Cs=self.Cs,
                cv=self.cv,
                penalty="l2",
                solver=self.linear_solver,
                max_iter=self.max_iter,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            )
            linear_probe.fit(X_train, y_train)

            # Make predictions
            predictions = linear_probe.predict(X_test)
            predictions_score = linear_probe.predict_proba(X_test)

            # Check arrays
            X_train, y_train = check_array(X_train), check_array(y_train)
            X_test, y_test = check_array(X_test), check_array(y_test)

            # Compute classification metrics
            metrics_report = classification_report(y_test, predictions, output_dict=True)
            metrics_report["ROC-AUC"] = roc_auc_score(y_test, predictions_score)

            # Log the results
            if log:
                base_name = f"logistic_probe_{self.dataset_name}" \
                    if self.dataset_name is not None else "logistic_probe"
                for name, value in metrics_report.items():
                    if isinstance(value, dict):
                        solver.log_dict(f"{base_name}/{name}", value)
                    else:
                        solver.log(f"{base_name}/{name}", value)
            
            # Return all metrics
            return metrics_report

    def on_validation_epoch_end(self, solver):
        if self.frequency == "by_epoch":
            self.linear_probing(solver)

    def on_fit_end(self, solver):
        if self.frequency == "by_fit":
            self.linear_probing(solver)


class KNeighborsClassifierCVCallback(Callback):
    """Performs KNN classification on top of a `torch.nn.Module` 
    trained by a `neuroclav.solvers.base.Solver`.

    Concretely, after each validation loop or each fit, this callback:
        1) Embeds the input data through the torch model
        2) Performs n-fold CV to find the best `n_neighbors` neighbors
        3) Log the main classification metrics including:
            * precision, recall, f1-score, support for each class + averaged overall 
            * global accuracy
            * ROC-AUC 
    """
    
    def __init__(self, 
                 train_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 dataset_name: Union[str, None]=None,
                 n_neighbors: List[int]=[2, 5, 10],
                 cv: int=5,
                 n_jobs: int=None,
                 scoring: str="balanced_accuracy",
                 frequency: str = "by_epoch",
                 **extraction_kwargs):
        """
        Parameters
        ----------
        train_dataloader: torch.utils.data.DataLoader
            Training dataloader to use for fitting the classifier. 

        test_dataloader: torch.utils.data.DataLoader
            Testing dataloader to use for evaluating the classifier. 
        
        dataset_name: str
            Name of the dataset to be used for classification. This name
            will appear when logging the results as `knn_classification_probe_<dataset_name>`.
            If None, only `knn_classification_probe` is displayed.
        
        n_neighbors : list of int, default=[2, 5, 10]
            Arrays of `n_neighbors` values to try in CV. 
            It corresponds to the number of neighbors to use by the KNN on the training set.

        cv: int or cross-validation generator, default=5
            How many folds to use for cross-validating the `C` regularization strenght 
            in the `LogisticRegression`. 

        n_jobs: int, default=None
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. 
        
        scoring: str in {"accuracy", "balanced_accuracy", "f1", ...}, default="balanced_accuracy" 
            Which scoring function to use to cross-validate the `C` hyper-parameter.
            For a complete list of scoring options, check 
            https://scikit-learn.org/1.4/modules/model_evaluation.html#scoring
        
        frequency: str in {'by_epoch', 'by_fit'}
            When to apply the linear probing. Either 'by_epoch', after each validation loop 
            or 'by_fit', only once at the end of training.

        extraction_kwargs: dict
            Keyword arguments given to `solver.get_embedding` method
        """
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.dataset_name = dataset_name
        self.n_neighbors = n_neighbors
        self.cv = cv
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.frequency = frequency
        self.extraction_kwargs = extraction_kwargs

    def knn_probing(self, solver, log: bool=True):
            
            if not hasattr(solver, "get_embedding"):
                raise ValueError("`get_embedding` must be implemented for linear probing")
            # Embed the data
            X_train, y_train = solver.get_embedding(self.train_dataloader, **self.extraction_kwargs)
            X_test, y_test = solver.get_embedding(self.test_dataloader, **self.extraction_kwargs)

            # Check arrays
            X_train, y_train = check_array(X_train), check_array(y_train)
            X_test, y_test = check_array(X_test), check_array(y_test)

            # Fit the classifier
            linear_probe = GridSearchCV(
                KNeighborsClassifier(),
                param_grid={"n_neighbors": self.n_neighbors},
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                cv=self.cv)
            linear_probe.fit(X_train, y_train)

            # Make predictions
            predictions = linear_probe.predict(X_test)
            predictions_score = linear_probe.predict_proba(X_test)
            
            # Check arrays
            X_train, y_train = check_array(X_train), check_array(y_train)
            X_test, y_test = check_array(X_test), check_array(y_test)

            # Compute classification metrics
            metrics_report = classification_report(y_test, predictions, output_dict=True)
            metrics_report["ROC-AUC"] = roc_auc_score(y_test, predictions_score)

            # Log the results
            if log:
                base_name = f"knn_classification_probe_{self.dataset_name}" \
                    if self.dataset_name is not None else "knn_classification"
                for name, value in metrics_report.items():
                    if isinstance(value, dict):
                        solver.log_dict(f"{base_name}/{name}", value)
                    else:
                        solver.log(f"{base_name}/{name}", value)
            
            # Return all metrics
            return metrics_report

    def on_validation_epoch_end(self, solver):
        if self.frequency == "by_epoch":
            self.knn_probing(solver)

    def on_fit_end(self, solver):
        if self.frequency == "by_fit":
            self.knn_probing(solver)
