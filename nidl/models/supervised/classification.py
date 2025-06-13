from pytorch_lightning.utilities.types import LRSchedulerPLType
from torch.optim import Optimizer
import torch.nn as nn
from typing import  Dict, Any, Union, Optional, Type, Sequence, Tuple, List
import torch
import logging
# This library
from nidl.models.base import BaseEstimator
from nidl.utils.validation import check_is_fitted
from nidl.volume.backbones import AlexNet, resnet18, resnet50, densenet121, \
    VisionTransformer, Offset0ModelMSE, MLP, Linear


class DeepClassifier(BaseEstimator):
    """Supervised deep model for classification tasks.

    Input data are mapped to predictions (class labels) with a neural network.
    The model optimizes a binary cross-entropy loss (default) or other classification losses. It can be
    trained on a very large dataset (e.g. UKB with 100k subjects and 3D volumes of >1M voxels).
    This nidl estimator has `fit` and `predict` methods implemented.

    NB: this model can be used for linear classification as well by defining a linear encoder.

    """
    def __init__(self, 
                 encoder: Union[str, nn.Module, Type[nn.Module]]="alexnet_3d", 
                 encoder_kwargs: Optional[Dict[str, Any]]=dict(in_channels=1, n_embedding=1),
                 loss: Union[str, nn.Module]='bce_with_logits',
                 optimizer: Union[str, Optimizer, Type[Optimizer]]="sgd", 
                 optimizer_kwargs: Optional[Dict[str, Any]]=dict(momentum=0.9, weight_decay=1e-4),
                 learning_rate: float=1e-3,
                 lr_scheduler: Optional[Union[str, LRSchedulerPLType, Type[LRSchedulerPLType]]]="constant",
                 lr_scheduler_kwargs: Optional[Dict[str, Any]]=None,
                 **kwargs: Any
                ):
        """
        Parameters
        ----------
        encoder: str in {'alexnet_3d', 'resnet18_3d', 'resnet50_3d', 'densenet121_3d', 'mlp', 
                         'cebra', 'vit', 'linear'} 
                or nn.Module or class, default='alexnet_3d'
            Which DNN architecture to use for predicting the targets. 
            If not in the default backbones, a PyTorch :class:`~torch.nn.Module` is expected. 
            In general, the uninstantiated class should be passed, although instantiated
            modules will also work.

        encoder_kwargs: dictionary or None, default=dict(in_channels=1, n_embedding=1)
            It specifies the options for building the encoder (depends on each architecture).
            By default, it builds an AlexNet with 1 input channel and output dimension 1 (for binary classification).
             Examples: 
                * encoder='alexnet_3d', encoder_kwargs={"in_channels": 1, "n_embedding": n_classes}
                    builds an AlexNet with 1 input channel (default) and output dimension 'n_classes'.
                * encoder='linear', encoder_kwargs={"in_features": 10, "out_features": n_classes, "bias": True}
                    builds a linear layer with input dimension 10 and output dimension 'n_classes'.
                * encoder='mlp', encoder_kwargs={"layers": [10, 4, 3, 2], "n_embedding": n_classes} 
                    builds an MLP with 4 hidden layers, the input dimension being 10. 
                * encoder='cebra', encoder_kwargs={"num_input": 10, "n_embedding": n_classes} 
                    builds an MLP with input dimension 10 (adapted from CEBRA)

        loss: str in {"bce_with_logits", "bce", "cross_entropy_loss"} or nn.Module, default="bce_with_logits"
            Loss to optimize. By default, Binary Cross-Entropy loss is optimized for binary classification.
            If a string is provided, it will be mapped to the corresponding PyTorch loss function.

        optimizer: str in {'sgd', 'adam', 'adamW'} or Optimizer or class, default='sgd'
            The optimizer to use for training the model. It can be a string:
            'sgd': stochastic gradient descent (with eventually momentum)
            'adam': stochastic first order gradient-based optimizer
            'adamW' (default): Adam optimizer with decoupled weight decay regularization 
               (see ''Decoupled Weight Decay Regularization, Loshchilov and Hutter, ICLR 2019'')
            or an instantiated object of type `torch.optim.Optimizer` or a class inheriting from it.
        
        optimizer_kwargs: dictionary or None, default={'momentum': 0.9, 'weight_decay': 1e-4}
            Arguments to give to optimizer ('sgd' by default) where 'momentum' and 'weight_decay' are the
            scaling factor for l2 penalization on model's weights and the momentum factor for SGD, respectively.
            This is ignored if `optimizer` is an instantiated object.

        learning_rate: float, default=1e-3
            The initial learning rate used.

        lr_scheduler: str in {'cosine_annealing', 'constant'} or 
            LRSchedulerPLType or class or None, default='constant'
            The learning rate scheduler to use.

        lr_scheduler_kwargs: dictionary or None, default=None
            Additional keyword arguments for the learning rate scheduler.
            By default, it sets the minimum learning rate to 5e-4 for the cosine annealing scheduler.
            If `lr_scheduler` is an instantiated object, these kwargs are ignored.
        
        **kwargs: dict, optional
            Additional keyword arguments for the BaseEstimator class, such as `max_epochs`, `max_steps`,
            `num_sanity_val_steps`, `check_val_every_n_epoch`, `callbacks`, etc.

        Attributes
        ----------
        encoder_: torch.nn.Module
            Deep neural network trained to map input data to predictions.
         
        loss_: nn.Module
            The loss function used for training the model.
        """
        super().__init__(**kwargs)
        self.encoder = encoder
        self.encoder_kwargs = encoder_kwargs if encoder_kwargs is not None else {}
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs if lr_scheduler_kwargs is not None else {}
        self.optimizer_kwargs = optimizer_kwargs
        self.save_hyperparameters()


    def fit(self, train_dataloader, val_dataloader=None):
        """ 
        Fit the model on training data. 

        Parameters
        ----------
        train_dataloader: torch DataLoader
            DataLoader for the training dataset.

        val_dataloader: torch DataLoader, optional
            DataLoader for the validation dataset.

        Returns
        ----------
        self: DeepClassifier
            The fitted model.

        """
        # Instantiate the encoder + loss
        self.encoder_ = self._build_encoder(self.encoder, self.encoder_kwargs)
        self.loss_ = self._build_loss(self.loss)

        # Fit the model
        return super().fit(train_dataloader, val_dataloader)


    def predict(self, dataloader):
        check_is_fitted(self, "encoder_")
        return super().predict(dataloader)


    def predict_step(self,  batch: Any, batch_idx: int):
        """ Computes the predictions for the given batch. 
        This method is called during predict.

        Parameters
        ----------
        batch: Any
            A batch of data that has been generated from val_dataloader.
            It should be a pair of Tensors (X, y) where X is the input
            and y is the (continuous) label.
        
        batch_idx: int
            The index of the current batch.
        """
        X, y = self.parse_batch(batch)
        return self.forward(X)
    
    
    def predict_epoch_end(self, outputs: List[torch.Tensor]):
        """ Aggregate outputs from 'predict_step' in a consistent torch.Tensor
        
        Parameters
        ----------
        outputs: List[torch.Tensor]
            Outputs given by 'predict_step'.
        
        Returns
        ----------
        torch.Tensor
            The concatenated outputs along the batch dimension.
        """
        return torch.cat(outputs, dim=0)

    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ 
        Forward pass through the encoder to obtain the predictions.

        Parameters
        ----------
        X: torch.Tensor
            Input data tensor.

        Returns
        -------
        predictions: torch.Tensor
            The output predictions.
        """
        X = X.to(self.device)
        y_pred = self.encoder_(X)
        return y_pred
    

    def training_step(self,  batch: Any, batch_idx: int):
        """ Perform one training step during an epoch and computes training loss.

        Parameters
        ----------
        batch: Any 
            A batch of data that has been generated from train_dataloader.
            It should be a pair of Tensors (X, y) where X is the input
            and y is the categorical label.

        batch_idx: int
            The index of the current batch.
        
        Returns
        ----------
        {"loss": torch.Tensor, "y_pred": torch.Tensor, "y_true": torch.Tensor}
            Training loss, predictions and ground truth computed on this batch.
        """
        X, y = self.parse_batch(batch)
        y_pred = self.encoder_(X)
        loss = self.loss_(y_pred, y)
        self.log("loss/train", loss, prog_bar=True)
        return {"loss": loss, 
                "y_pred": y_pred.cpu().detach(), 
                "y_true": y.cpu().detach()}
    

    def validation_step(self,  batch: Any, batch_idx: int):
        """ Only computes the validation embedding for further metrics computation.

        Parameters
        ----------
        batch: Any
            A batch of data that has been generated from val_dataloader.
            It should be a pair of Tensors (X, y) where X is the input
            and y is the (continuous) label.
        
        batch_idx: int
            The index of the current batch.

        Returns
        ----------
        {"loss": torch.Tensor, "y_pred": torch.Tensor, "y_true": torch.Tensor}
            Training loss, predictions and ground truth computed on this batch.
        """

        X, y = self.parse_batch(batch)
        y_pred = self.encoder_(X)
        loss = self.loss_(y_pred, y)
        self.log("loss/val", loss)
        return {"loss": loss, 
                "y_pred": y_pred.cpu().detach(), 
                "y_true": y.cpu().detach()}


    def parse_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Parses the batch to return (X, y)

        Parameters
        ----------
        batch: Any
            A batch of data that has been generated from train_dataloader or val_dataloader.
            It should be a pair of Tensors (X, y) where X is the input
            and y is the (continuous) label.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            Tuple (X, y)
        """
        if isinstance(batch, Sequence) and len(batch) == 2:
            return batch[0].to(self.device), batch[1].to(self.device)
        elif isinstance(batch, torch.Tensor) and len(batch) == 2:
            return batch[0].to(self.device), batch[1].to(self.device)
        else:
            raise ValueError("batch should be a tuple of 2 " \
            "Tensors (representing input and label), got %s" % type(batch))
        

    def configure_optimizers(self):
        known_optimizers = {
            "adam": torch.optim.Adam,
            "adamW": torch.optim.AdamW,
            "sgd": torch.optim.SGD
        }
        if isinstance(self.optimizer, str):
            if self.optimizer not in known_optimizers:
                raise ValueError(f"Optimizer '{self.optimizer}' is not implemented. "
                                 f"Please use one of the available optimizers: "
                                 f"{', '.join(known_optimizers.keys())}")
            optimizer = known_optimizers[self.optimizer](
                params=self.encoder_.parameters(),
                lr=self.learning_rate,
                **self.optimizer_kwargs
            )
        elif isinstance(self.optimizer, Optimizer):
            if len(self.optimizer_kwargs) > 0:
                logging.getLogger(__name__).warning("optimizer is already instantiated, ignoring 'optimizer_kwargs'")
            optimizer = self.optimizer
        elif isinstance(self.optimizer, type) and issubclass(self.optimizer, Optimizer):
            optimizer = self.optimizer(
                params=self.encoder_.parameters(),
                lr=self.learning_rate,
                **self.optimizer_kwargs
            )
        else:
            raise ValueError(f"Optimizer must be a string, a PyTorch Optimizer, or a class "
                                      f"inheriting from Optimizer, got {type(self.optimizer)}")
        if self.lr_scheduler is None:
            scheduler = None
        elif isinstance(self.lr_scheduler, str):
            known_schedulers = {
                "cosine_annealing": torch.optim.lr_scheduler.CosineAnnealingLR,
                "constant": None
            }
            if self.lr_scheduler not in known_schedulers:
                raise ValueError(f"Learning rate scheduler '{self.lr_scheduler}' is not implemented. "
                                 f"Please use one of the available schedulers: "
                                 f"{', '.join(known_schedulers.keys())}")
            if self.lr_scheduler == "linear_warmup_cosine_annealing":
                scheduler = known_schedulers[self.lr_scheduler](
                    optimizer=optimizer,
                    max_epochs=self.trainer.max_epochs,
                    **self.lr_scheduler_kwargs
                )
            elif self.lr_scheduler == "cosine_annealing":
                scheduler = known_schedulers[self.lr_scheduler](
                    optimizer=optimizer,
                    T_max=self.trainer.max_epochs,
                    **self.lr_scheduler_kwargs
                )
            elif self.lr_scheduler == "constant":
                scheduler = None
        elif isinstance(self.lr_scheduler, LRSchedulerPLType):
            if len(self.lr_scheduler_kwargs) > 0:
                logging.getLogger(__name__).warning("lr_scheduler is already instantiated, ignoring 'lr_scheduler_kwargs'")
            scheduler = self.lr_scheduler
        elif isinstance(self.lr_scheduler, type) and issubclass(self.lr_scheduler, LRSchedulerPLType):
            scheduler = self.lr_scheduler(
                optimizer=optimizer,
                **self.lr_scheduler_kwargs
            )
        else:
            raise ValueError(f"Learning rate scheduler must be None, a string, a PyTorch LRSchedulerPLType, "
                             f"or a class inheriting from LRSchedulerPLType, got {type(self.lr_scheduler)}")
        if scheduler is None:
            return optimizer
        else:
            return [optimizer], [scheduler]


    def _build_encoder(self,
                       encoder: Union[str, nn.Module, Type[nn.Module]], 
                       encoder_kwargs: Dict[str, Any]) -> nn.Module:
        known_encoders = {
            "alexnet_3d": AlexNet,
            "resnet18_3d": resnet18,
            "resnet50_3d": resnet50,
            "densenet121_3d": densenet121,
            "mlp": MLP,
            "linear": Linear,
            "cebra": Offset0ModelMSE,
            "vit": VisionTransformer
        }

        if isinstance(encoder, str):
            if encoder not in known_encoders:
                raise ValueError(f"Encoder '{encoder}' is not implemented. "
                                 f"Please use one of the available encoders: "
                                 f"{', '.join(known_encoders.keys())}")
            encoder = known_encoders[encoder](**encoder_kwargs)
        elif isinstance(encoder, nn.Module):
            if encoder_kwargs is not None and len(encoder_kwargs) > 0:
                logging.getLogger(__name__).warning("encoder is already instantiated, ignoring 'encoder_kwargs'")
        elif isinstance(encoder, type) and issubclass(encoder, nn.Module):
            encoder = encoder(**encoder_kwargs)
        else:
            raise ValueError(f"Encoder must be a string, a PyTorch nn.Module, or a class "
                                      f"inheriting from nn.Module, got {type(encoder)}")
        return encoder


    def _build_loss(self, loss: Union[str, nn.Module]) -> nn.Module:
        """ Builds the regression loss function.

        Parameters
        ----------
        loss: str in {'cross_entropy', 'bce_with_logits', 'bce'} or nn.Module
            Which loss function to use for classification.
        Returns
        -------
        loss_fn: nn.Module
            The PyTorch loss function to use for training.
        """
        if isinstance(loss, nn.Module):
            return loss
        if loss == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif loss == "bce_with_logits":
            return nn.BCEWithLogitsLoss()
        elif loss == "bce":
            return nn.BCELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss}")

