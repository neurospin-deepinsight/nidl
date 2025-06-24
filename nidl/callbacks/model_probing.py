from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, mean_absolute_error, \
    root_mean_squared_error, r2_score
from scipy.stats import pearsonr
from typing import List, Union, Any, Tuple, Sequence
import torch
from pytorch_lightning import Callback
from abc import ABC, abstractmethod
# Local import
from nidl.utils.validation import check_array
from nidl.estimators import BaseEstimator


class ModelProbing(ABC, Callback):
    """ Define the basic logic of model's probing:

        1) Embeds the input data (training+test) through the BaseEstimator 
           (calling .forward() by default)
        2) Train the probe on the training embedding
        3) Test the probe on the test embedding and log the metrics
    
    # TODO: be more flexible on the frequency (could be every 'n' validation loop)
    # TODO: include a potential validation set for training the probe

    """
    def __init__(
               self,
               train_dataloader: DataLoader,
               test_dataloader: DataLoader,
               frequency: str="by_epoch",
               prog_bar: bool=True
               ):
        """
        Parameters
        ----------
        train_dataloader: torch.utils.data.DataLoader
            Training dataloader to use for fitting the ridge regression. 

        test_dataloader: torch.utils.data.DataLoader
            Testing dataloader to use for evaluating the ridge regression. 

        frequency: str in {'by_epoch', 'by_fit'}
            When to apply the linear probing. Either 'by_epoch', after each validation loop 
            or 'by_fit', only once at the end of training.
        
        prog_bar: bool, default=True
            Whether to display the metrics in the progress bar.
        """
        super().__init__()
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.frequency = frequency
        self.prog_bar = prog_bar

    @abstractmethod
    def fit(self, X, y):
        """Fit the probe on (X, y)"""
        pass

    @abstractmethod
    def predict(self, X):
        """Predict on new data X"""
        pass

    @abstractmethod
    def log_metrics(self, pl_module, y_pred, y_true):
        """Log the metrics"""
        pass

    def linear_probing(self, pl_module: BaseEstimator, log: bool=True):
            
        if not isinstance(pl_module, BaseEstimator):
            raise ValueError("Your Lightning module must derive from 'BaseEstimator', " \
                             "got %s" % type(pl_module))
        
        # Embed the data
        X_train, y_train = self.extract_features(pl_module, self.train_dataloader)
        X_test, y_test = self.extract_features(pl_module, self.test_dataloader)

        # Check arrays
        X_train, y_train = check_array(X_train), check_array(y_train, ensure_2d=False)
        X_test, y_test = check_array(X_test), check_array(y_test, ensure_2d=False)

        # Fit the probe
        self.fit(X_train, y_train)

        # Make predictions
        y_pred = self.predict(X_test)

        # Compute/Log metrics
        self.log_metrics(pl_module, y_pred, y_test)
    

    def extract_features(self, pl_module, dataloader):
        """Extract features from a given dataloader with the current BaseEstimator.
        By default, it uses the forward() logic applied on each batch to get the embeddings.
        
        !! You may override this method for your specific need.
        """

        is_training = pl_module.training  # Save state

        pl_module.eval()
        X, y = [], []

        with torch.no_grad():
            for batch in dataloader:
                x_batch, y_batch = self.parse_batch(batch)
                x_batch = x_batch.to(pl_module.device)
                features = pl_module.forward(x_batch)
                X.append(features.detach().cpu())
                y.append(y_batch.detach().cpu())
        X = torch.cat(X).numpy()
        y = torch.cat(y).numpy()

        if is_training:
            pl_module.train()

        return X, y 
    

    def parse_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Parses the batch to return (X, y)

        Parameters
        ----------
        batch: Any
            A batch of data that has been generated from train_dataloader or test_dataloader.
            It should be a pair of Tensors (X, y) where X is the input
            and y is the (continuous) label.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            Tuple (X, y)
        """
        if isinstance(batch, Sequence) and len(batch) == 2:
            return batch[0], batch[1]
        elif isinstance(batch, torch.Tensor) and len(batch) == 2:
            return batch[0], batch[1]
        else:
            raise ValueError("batch should be a tuple of 2 " \
            "Tensors (representing input and label), got %s" % type(batch))        
    

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.frequency == "by_epoch":
            self.linear_probing(pl_module)


    def on_fit_end(self, trainer, pl_module):
        if self.frequency == "by_fit":
            self.linear_probing(pl_module)



class RidgeCVCallback(ModelProbing):
    """Perform Ridge regression on top of a `nidl.models.base.BaseEstimator` model.
      
      Concretely, after each validation loop or each fit, this callback:
        1) Embeds the input data through the BaseEstimator (calling .forward() by default)
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
               prog_bar: bool=True
               ):
        """
        Parameters
        ----------
        train_dataloader: torch.utils.data.DataLoader
            Training dataloader to use for fitting the ridge regression. 

        test_dataloader: torch.utils.data.DataLoader
            Testing dataloader to use for evaluating the ridge regression. 
        
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
        
       prog_bar: bool, default=True
            Whether to display the metrics in the progress bar.
        """
        super().__init__(train_dataloader=train_dataloader, 
                         test_dataloader=test_dataloader, 
                         frequency=frequency, prog_bar=prog_bar)
        self.dataset_name = dataset_name
        self.alphas = alphas
        self.cv = cv
        self.scoring = scoring
        self.probe = RidgeCV(
            alphas=self.alphas,
            cv=self.cv,
            scoring=self.scoring,
        )
    
    def fit(self, X, y):
        return self.probe.fit(X, y)
    
    def predict(self, X):
        return self.probe.predict(X)
    
    def log_metrics(self, pl_module, y_pred, y_true):
         # Compute regression metrics
        metrics_report = dict()
        metrics_report["MAE"] = mean_absolute_error(y_true, y_pred)
        metrics_report["RMSE"] = root_mean_squared_error(y_true, y_pred)
        metrics_report["R2"] = r2_score(y_true, y_pred)
        metrics_report["pearson_r"] = pearsonr(y_true.flatten(), y_pred.flatten()).statistic

        # Log the results
        base_name = f"ridge_probe_{self.dataset_name}" if self.dataset_name is not None else "ridge_probe"
        for name, value in metrics_report.items():
            pl_module.log(f"{base_name}/{name}", value, prog_bar=self.prog_bar, on_epoch=True)
    

class KNeighborsRegressorCVCallback(Callback):
    """Perform KNN regression on top of a `nidl.models.base.BaseEstimator` model.

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
               prog_bar: bool=True):
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

        prog_bar: bool, default=True
            Whether to display the metrics in the progress bar.
        """
        super().__init__(train_dataloader=train_dataloader, 
                         test_dataloader=test_dataloader, 
                         frequency=frequency, prog_bar=prog_bar)
        self.dataset_name = dataset_name
        self.n_neighbors = n_neighbors
        self.cv = cv
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.probe = GridSearchCV(
            KNeighborsRegressor(),
            param_grid={"n_neighbors": self.n_neighbors},
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            cv=self.cv
        )
    
    def fit(self, X, y):
        return self.probe.fit(X, y)
    
    def predict(self, X):
        return self.probe.predict(X)
    
    def log_metrics(self, pl_module, y_pred, y_true):
         # Compute regression metrics
        metrics_report = dict()
        metrics_report["MAE"] = mean_absolute_error(y_true, y_pred)
        metrics_report["RMSE"] = root_mean_squared_error(y_true, y_pred)
        metrics_report["R2"] = r2_score(y_true, y_pred)
        metrics_report["pearson_r"] = pearsonr(y_true.flatten(), y_pred.flatten()).statistic

        # Log the results
        base_name = f"knn_regression_probe_{self.dataset_name}" if self.dataset_name is not None else "knn_regression_probe"
        for name, value in metrics_report.items():
            pl_module.log(f"{base_name}/{name}", value, prog_bar=self.prog_bar, on_epoch=True)


class LogisticRegressionCVCallback(Callback):
    """Performs logistic regression (classification) on top of a `nidl.models.base.BaseEstimator` model.

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
                 prog_bar: bool=True):
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

        prog_bar: bool, default=True
            Whether to display the metrics in the progress bar.
        """
        super().__init__(train_dataloader=train_dataloader, 
                         test_dataloader=test_dataloader, 
                         frequency=frequency, prog_bar=prog_bar)
        self.dataset_name = dataset_name
        self.Cs = Cs
        self.cv = cv
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.linear_solver = linear_solver
        self.probe = LogisticRegressionCV(
                Cs=self.Cs,
                cv=self.cv,
                penalty="l2",
                solver=self.linear_solver,
                max_iter=self.max_iter,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
        )

    def fit(self, X, y):
        return self.probe.fit(X, y)
    
    def predict(self, X):
        return (self.probe.predict(X), self.probe.predict_proba(X))
    
    def log_metrics(self, pl_module, y_pred, y_true):
        # Compute classification metrics
        metrics_report = classification_report(y_true, y_pred[0], output_dict=True)
        metrics_report["ROC-AUC"] = roc_auc_score(y_true, y_pred[1])

        # Log the results
        base_name = f"logistic_probe_{self.dataset_name}" \
            if self.dataset_name is not None else "logistic_probe"
        for name, value in metrics_report.items():
            if isinstance(value, dict):
                pl_module.log_dict(f"{base_name}/{name}", value)
            else:
                pl_module.log(f"{base_name}/{name}", value)


class KNeighborsClassifierCVCallback(Callback):
    """Performs KNN classification on top of a `nidl.models.base.BaseEstimator` model.

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
                 prog_bar: bool=True):
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

        prog_bar: bool, default=True
            Whether to display the metrics in the progress bar.
        """
        super().__init__(train_dataloader=train_dataloader, 
                         test_dataloader=test_dataloader, 
                         frequency=frequency, prog_bar=prog_bar)
        self.dataset_name = dataset_name
        self.n_neighbors = n_neighbors
        self.cv = cv
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.frequency = frequency
        self.prog_bar = prog_bar
        self.probe = GridSearchCV(
                KNeighborsClassifier(),
                param_grid={"n_neighbors": self.n_neighbors},
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                cv=self.cv
        )

    def fit(self, X, y):
        return self.probe.fit(X, y)
    
    def predict(self, X):
        return (self.probe.predict(X), self.probe.predict_proba(X))
    
    def log_metrics(self, pl_module, y_pred, y_true):
        # Compute classification metrics
        metrics_report = classification_report(y_true, y_pred[0], output_dict=True)
        metrics_report["ROC-AUC"] = roc_auc_score(y_true, y_pred[1])

        # Log the results
        base_name = f"knn_classification_probe_{self.dataset_name}" \
            if self.dataset_name is not None else "knn_classification_probe"
        for name, value in metrics_report.items():
            if isinstance(value, dict):
                pl_module.log_dict(f"{base_name}/{name}", value)
            else:
                pl_module.log(f"{base_name}/{name}", value)