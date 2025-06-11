import logging
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted
import torch
import pickle, os
import numpy as np
from torch.nn import Sequential, Linear, MSELoss, L1Loss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from typing import Dict, Any, List, Union, Optional
from collections import OrderedDict
import yaml
# This library
from neuroclav.data.split import get_split_datasets, get_dataloader, get_dataset
from neuroclav.data.split import ValidSplit
from neuroclav.solvers.regression_solver import RegressionSolver
from neuroclav.callbacks.base import Callback
from neuroclav.models.resnet import resnet18
from neuroclav.models.densenet import densenet121
from neuroclav.models.alexnet import AlexNet
from neuroclav.models.deepbrainnet import DeepBrainNet
from neuroclav.models.cebra import Offset0ModelMSE
from neuroclav.models.mlp import MLP
from neuroclav.utils.util import get_labels_from_dataset
from neuroclav.data.collate import BaseCollateFunction
from neuroclav.transforms.transform import Transform
from neuroclav.transforms.preprocessing.spatial.volume_transform import VolumeTransform


class DeepRegressor(BaseEstimator, TransformerMixin, RegressorMixin):
    """Supervised deep model for regression on neuroimaging data.

    Input data are mapped to an embedding with a deep neural network. Targets are
    predicted with a linear layer on top of the embedding. 
    The model optimizes a Mean Squared Error (MSE) loss (default) or l1 loss. It can be 
    trained on a very large dataset (e.g. UKB with 100k subjects and 3D volumes of >1M voxels).

    This scikit-learn estimator has `fit`, `predict` and `transform` methods implemented.

    Examples
    --------
    >>> from manifold.supervised import DeepRegressor
    >>> from data.datasets.openbhb import OpenBHB
    >>> regressor = DeepRegressor(encoder="resnet18_3d", n_components=8)
    
    Example 1: OpenBHB
    >>> openBHB_train = OpenBHB("/neurospin/hc/openBHB", train=True, target="age")
    >>> openBHB_test = OpenBHB("/neurospin/hc/openBHB", train=False, taget="age")
    >>> regressor.fit(openBHB_train)
    >>> y_test = regressor.predict(openBHB_test)
    >>> test_embed = regressor.transform(openBHB_test)

    """

    def __init__(self, 
                 encoder: str="alexnet_3d", 
                 encoder_kwargs: Optional[dict]=None,
                 n_components: int=16,
                 train_transform: Transform=VolumeTransform(),
                 val_transform: Transform=VolumeTransform(),
                 test_transform: Transform=VolumeTransform(),
                 loss: str="mse",
                 optimizer: str="adam",
                 callbacks: Union[List[Callback], Callback, None]=None,
                 train_split : Union[None, callable]=ValidSplit(5),
                 learning_rate: str='constant', 
                 learning_rate_init: float=1e-4,
                 power_t=0.9, 
                 max_iteration: int=100, 
                 batch_size: int=32, 
                 valid_frequency=10,
                 device: str="cuda_if_available", 
                 root_dir: str=os.getcwd(),
                 exp_name: str="regression_logs",
                 random_state: int=None,
                 optimizer_kwargs: Dict[str, Any]=dict(betas=(0.9, 0.99), weight_decay=5e-5),
                 num_workers: int=0, 
                 logging_level: int=logging.INFO
                ):
        """
        Parameters
        ----------

        encoder: {'alexnet_3d', 'resnet18_3d', 'densenet121_3d', 'deepbrainnet', 'mlp', 'offset'}, 
                default='alexnet'
            Which DNN architecture to use for encoding the input. 

        encoder_kwargs: dictionary, default=None
            It specifies the options for building the encoder (depends on each architecture).
             Examples: 
                * encoder='mlp', encoder_kwargs={"layers": [10, 4, 3, 2]} builds an MLP with 4 hidden layers, 
                    the input dimension being 10. Output dimension is always 'n_components'.
                * encoder='offset', encoder_kwargs={"num_input": 10} builds an MLP with input dimension 10 (adapted from CEBRA)
                * encoder='deepbrainnet', encoder_kwargs={"input_size": (1, 128, 128, 128)} builds a DeepBrainNet model.
        
        train_transform: neuroclav.data.transforms.base.Transform, default=VolumeTransform()
            Training transformation applied when training the model.
        
        val_transform: neuroclav.data.transforms.base.Transform, default=VolumeTransform()
            Validation transformation applied when validating the model.
        
        test_transform: neuroclav.data.transforms.base.Transform, default=VolumeTransform()
            Test transformation applied when transforming the data.
        
        n_components: int, default=16
            Dimension of the embedded space (before regression on the final targets).
        
        loss: str in {"mse", "l1"}, default="mse"
            Loss to optimize. 

        optimizer: {'sgd', 'adam'}, default='adam'
            'sgd': stochastic gradient descent (with eventually momentum)
            'adam': stochastic first order gradient-based optimizer
        
        callbacks: List[Callback], Callback or None, default=None    
            Callback(s) to be called during training/validation of the model. 
            If None, no callbacks are used. 
        
        train_split : None or callable, default=neuroclav.data.dataset.ValidSplit(5, stratified=True)
            If ``None``, there is no train/validation split. Else, ``train_split``
            should be a function or callable that is called with X and y
            data and should return the tuple ``dataset_train, dataset_valid``.
            The validation data may be ``None``.

        learning_rate: {'constant', 'invscaling'}, default='constant'
            Learning rate schedule for weight updates.
            * 'constant' is a constant learning rate given by 'learning_rate_init'.
            * 'invscaling' gradually decreases the learning rate at each time step 't'
               using an inverse scaling exponent of 'power_t'.
               effective_learning_rate = learning_rate_init / pow(t, power_t)

        learning_rate_init: float, default=1e-4
            The initial learning weight used.

        power_t: float, default=0.9
            The exponent for inverse scaling learning rate. It is used in updating effective
            learning rate when the learning_rate is set to 'invscaling'.

        max_iteration: int, default=100
            Maximum number of iterations. The solver iterates until this number of iterations. 
            Please note that this determines the number of epochs
            (i.e. how many times each data point will be used), not the number of gradient steps
            
        batch_size: int, default=32
            Size of each mini-batch for stochastic optimizers.
            
        valid_frequency: Optional[int], default=None
            Frequency at which validation metrics are computed (in number of epochs)
            
        device: {'cpu', 'cuda', 'cuda_if_available'}, default='cuda_if_available'
            Device on which the model is trained. For 'cuda', all available gpu will be used.
            
        root_dir: str, os.PathLike, default=os.getcwd()
            The root directory in which all your experiments with different names and versions will be stored.
            Typically, this directory will contains model's weights and training/validation metrics.
        
        exp_name: str, default="regression_logs"
            Experiment name. This needs to be a string otherwise an Exception is raised.
            
        random_state: int, default=None
            Determines random number generation for model's initialization (`torch.manual_seed`),
            train-test split if `validation_fraction` is set and batch sampling. Pass an int for
            reproducible results across function calls.
            
        optimizer_kwargs: tuple, default=(('beta_1', 0.9), ('beta_2', 0.99), ('weight_decay', 5e-5))
            Arguments to give to optimizer ('adam' by default) where 'beta_1' and 'beta_2' are the
            exponential decay rate for estimates of first and second moment vector in Adam.
            'weight_decay' is scaling factor for l2 penalization on model's weights. It can be used for both
            'Adam' and 'SGD'
            
        num_workers: int, default 0
            Number of CPU workers used to pre-process data on-the-fly.
            
        logging_level: {0, 10, 20, 30, 40, 50}
            Logging level as defined by python

         Attributes
        ----------
        model_: torch nn.Module
            Deep neural network as regressor to predict target values.

        solver_: neuroclav.solvers.regression_solver.RegressionSolver
            Solver fitted on the data. 

        embedding_ : array-like of shape (n_samples, n_components)
            Stores the embedding vectors computed during call to `fit`.
        """
        
        self.encoder = encoder
        self.encoder_kwargs = encoder_kwargs if encoder_kwargs is not None else {}
        self.n_components = n_components
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.loss = loss
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.train_split = train_split
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iteration = max_iteration
        self.batch_size = batch_size
        self.valid_frequency = valid_frequency
        self.device = device
        self.root_dir = root_dir
        self.exp_name = exp_name
        self.random_state = random_state
        self.optimizer_kwargs = optimizer_kwargs
        self.num_workers = num_workers
        self.logging_level = logging_level


    def fit(self, X, y=None):
        """ Fit the model on data X and target(s) y.

        Parameters
        ----------
        X: PyTorch Dataset or array representing n-dimensional array (n_samples, *)

        y: array-like of shape (n_samples, n_targets) or (n_samples,), default=None
            Target values (real numbers). If None, the targets in 'X' are used instead.

        Returns
        ----------
        self: DeepRegressor
            The fitted model.

        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        self._validate_hyperparameters()

         # Split the datasets and get the dataloaders
        train_dataset, val_dataset = get_split_datasets(X, y, self.train_split)

        train_loader = get_dataloader(train_dataset, 
                                      collate_fn=BaseCollateFunction(self.train_transform),
                                      num_workers=self.num_workers,
                                      batch_size=self.batch_size,
                                      #multiprocessing_context=get_context("loky"),
                                      shuffle=True)
        
        val_loader = get_dataloader(val_dataset,
                                    collate_fn=BaseCollateFunction(self.val_transform),
                                    num_workers=self.num_workers,
                                    batch_size=self.batch_size,
                                    #multiprocessing_context=get_context("loky"),
                                    shuffle=False, 
                                    drop_last=False)
        
        # Get y to automatically compute the target shape
        y = get_labels_from_dataset(X, y=y, raise_if_none=True)

        # Initialize the model with random initialization + optimizer
        self.model_ = self._build_model(self.encoder, self.encoder_kwargs, num_targets=y.shape[1])
        optimizer, scheduler = self._get_optimizer()

        # Initialize the loss
        loss = self._get_loss(self.loss)

        # Intialize the Solver specific to this DeepRegressor estimator
        self.solver_ = RegressionSolver(
             model=self.model_, 
             loss=loss, 
             optimizer=optimizer, 
             scheduler=scheduler, 
             callbacks=self.callbacks, 
             max_iter=self.max_iteration,
             root_dir=self.root_dir,
             exp_name=self.exp_name,
             device=self.device, 
             logging_level=self.logging_level)
        
        # Save the parameters of this estimator before fitting the model.
        #TODO: also saves the !!data hparams!! used to fit the model.
        self.save_hyperparameters()

        # Fit the model to the data
        self.solver_.fit(
            train_loader, 
            val_loader,
            valid_frequency=self.valid_frequency)
        
        self.embedding_ = self.solver_.get_embedding(train_loader, as_tensor=False, return_y=False)
        return self


    def _validate_hyperparameters(self):
        if self.max_iteration <= 0:
            raise ValueError("max_iter must be > 0, got %s." % self.max_iteration)

        if (
                self.learning_rate in ["constant", "invscaling"]
                and self.learning_rate_init <= 0.0
        ):
            raise ValueError(
                "learning_rate_init must be > 0, got %s." % self.learning_rate
            )
        if self.learning_rate not in ["constant", "invscaling"]:
            raise ValueError("learning rate %s is not supported. " % self.learning_rate)
        supported_solvers = ["sgd", "adam"]
        if self.optimizer not in supported_solvers:
            raise ValueError(
                "The solver %s is not supported.  Expected one of: %s"
                % (self.optimizer, ", ".join(supported_solvers))
            )
    
    def _get_loss(self, loss: str):
        if loss == "mse":
            return MSELoss()
        elif loss == "l1":
            return L1Loss()

    def _get_optimizer(self):
        check_is_fitted(self, "model_")
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.model_.parameters(),
                                         lr=self.learning_rate_init,
                                         **self.optimizer_kwargs)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model_.parameters(),
                                        lr=self.learning_rate_init,
                                        **self.optimizer_kwargs)
        else:
            raise ValueError("Optimizer must be in {'adam', 'sgd'}, got %s"%self.optimizer)

        if self.learning_rate == "invscaling":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=self.power_t, step_size=1)
        elif self.learning_rate == "constant":
            scheduler = None
        else:
            raise ValueError("Learning rate must be in {'invscaling', 'constant'}, got %s"%self.learning_rate)
        return optimizer, scheduler

    def _build_model(self, encoder: str, encoder_kwargs: dict, num_targets=1):
        if encoder == "alexnet_3d":
            encoder_nn = AlexNet(n_embedding=self.n_components, **encoder_kwargs)
        elif encoder == "resnet18_3d":
            return resnet18(n_embedding=self.n_components, **encoder_kwargs)
        elif encoder == "densenet121_3d":
            encoder_nn = densenet121(n_embedding=self.n_components, **encoder_kwargs)
        elif encoder == 'mlp':
            encoder_nn = MLP(n_components=self.n_components, **encoder_kwargs)
        elif encoder == 'offset':
            encoder_nn = Offset0ModelMSE(n_embedding=self.n_components, **encoder_kwargs)
        elif encoder == "deepbrainnet":
            encoder_nn = DeepBrainNet(n_embedding=self.n_components, **encoder_kwargs)
        else:
            raise NotImplementedError()
        return Sequential(OrderedDict([("encoder", encoder_nn),
                                       ("regressor", Linear(self.n_components, num_targets))]))

    def transform(self, X, return_y: bool=False):
        """ Performs feedforward pass of X through the model and returns its embedding.
        Parameters
        ----------
        X: torch.utils.data.dataset.Dataset or array
            Dataset to embed.

        collate_fn: str or BaseCollateFunction, default "mri"
            Collate function to transform batch of data to tensor.

        return_y: bool, default False
            If True, also returns the "y" value in the input "X" Dataset.

        Returns
        ----------
        embedding or (embedding, y) if return_y==True: array or (array, array)
        """
        check_is_fitted(self, "solver_")

        X = get_dataset(X)
        loader = DataLoader(X, 
                            collate_fn=BaseCollateFunction(self.test_transform),
                            num_workers=self.num_workers,
                            batch_size=self.batch_size,
                            #multiprocessing_context=get_context("loky"),
                            shuffle=False)
        return self.solver_.get_embedding(loader, as_tensor=False, return_y=return_y)

    def predict(self, X: Dataset,
                collate_fn: Union[str, BaseCollateFunction]="base",
                return_y=False):
        """
        Predict target values from data in X. Eventually returns the true targets, if known
        
        Parameters
        ----------
        X: torch.utils.data.Dataset
            Dataset encoded by the model to output predictions.
        collate_fn: str in {'mri'} or neuroclav.data.collate.BaseCollateFunction, default='base'
            Collate function of the dataloader.
        return_y: boolean, default False
              If True, returns the true targets along with the predicted values
        
        Returns
        ----------
        y_pred or (y_pred, y_true) if return_y == True: array or (array, array)
        """
        check_is_fitted(self, "model_")
        collate_fn = self._get_collate_fn(collate_fn)
        loader = DataLoader(X, collate_fn=collate_fn,
                            num_workers=self.num_workers,
                            batch_size=self.batch_size,
                            #multiprocessing_context=get_context("loky"),
                            shuffle=False)
        return self.solver_.predict(loader, return_y=return_y)

    def save_hyperparameters(self):
        """Returns all the parameters used to build this Estimator. Eventual Callbacks are not saved."""
        check_is_fitted(self, "solver_")

        params = self.get_params()
        # Remove eventual callbacks in the params as they should **not** 
        # affect reproducility of the results
        if "callbacks" in params:
            del params["callbacks"]

        hname = "hparams.yaml"
        full_path = os.path.join(self.solver_.log_dir, hname)
        with open(full_path, "w") as file:
            yaml.dump(params, file)


    def save(self, filename: str=None):
        check_is_fitted(self, "solver_")
        if filename is None:
            filename = type(self).__name__
            
        with open(os.path.join(self.solver_.log_dir, filename), "wb") as f:
            pickle.dump(self, f)

