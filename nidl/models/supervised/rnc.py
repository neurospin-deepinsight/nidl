from pytorch_lightning.utilities.types import LRSchedulerPLType
from torch.optim import Optimizer
import torch.nn as nn
from typing import  Dict, Any, Union, Optional, Type, Sequence, Tuple
import torch
import numpy as np
import logging
# This library
from nidl.models.base import BaseEstimator, EmbeddingTransformerMixin
from nidl.volume.backbones import AlexNet, resnet18, resnet50, densenet121, \
    VisionTransformer, MLP, Offset0ModelMSE
from nidl.utils.lr_schedulers import LinearWarmupCosineAnnealingLR
from nidl.losses import RnCLoss


class RnC(BaseEstimator, EmbeddingTransformerMixin):
    """
    Rank-N-Contrast (RnC) [1] is a framework that learns continuous
    representations for regression by contrasting samples against each other based on
    their rankings in the target space. Authors demonstrate, theoretically and empirically,
    that RnC guarantees the desired order of learned representations in accordance with
    the target orders, enjoying not only better performance but also significantly im-
    proved robustness, efficiency, and generalization.

    [1] Rank-N-Contrast: Learning Continuous Representations for Regression, Zha et al., NeurIPS 2023
    """

    def __init__(self, 
                 encoder: Union[str, nn.Module, Type[nn.Module]]="alexnet_3d", 
                 encoder_kwargs: Optional[Dict[str, Any]]=None,
                 n_embedding: int=16,
                 temperature: float=2.0,
                 label_diff: str='l1',
                 optimizer: Union[str, Optimizer, Type[Optimizer]]="sgd", 
                 optimizer_kwargs: Optional[Dict[str, Any]]=dict(momentum=0.9, weight_decay=1e-4),
                 learning_rate: float=0.5,
                 lr_scheduler: Optional[Union[str, LRSchedulerPLType, Type[LRSchedulerPLType]]]="cosine_annealing",
                 lr_scheduler_kwargs: Optional[Dict[str, Any]]=dict(eta_min=5e-4),
                 **kwargs: Any
                ):
        """
        Parameters
        ----------
        encoder: str in {'alexnet_3d', 'resnet18_3d', 'resnet50_3d', 'densenet121_3d', 'mlp', 'cebra', 'vit'} 
                or nn.Module or class, default='alexnet_3d'
            Which DNN architecture to use for encoding the input. 
            If not in the default backbones, a PyTorch :class:`~torch.nn.Module` is expected. 
            In general, the uninstantiated class should be passed, although instantiated
            modules will also work.

        encoder_kwargs: dictionary or None, default=None
            It specifies the options for building the encoder (depends on each architecture).
            If `n_embedding` is specified here, it will override the `n_embedding` parameter.
             Examples: 
                * encoder='mlp', encoder_kwargs={"layers": [10, 4, 3, 2]} builds an MLP with 4 hidden layers, 
                    the input dimension being 10. Output dimension is always 'n_embedding'.
                * encoder='cebra', encoder_kwargs={"num_input": 10} builds an MLP with input dimension 10 (adapted from CEBRA)
        
        n_embedding: int, default=16
            Dimension of the embedding space.

        temperature: float, default=2.0
            The temperature parameter for the RnC loss function.
        
        label_diff: str in {'l1', 'l2'}, default='l1'
            Which distance to use between labels, ultimately used to rank 
            the samples according to this distance matrix in the loss function.

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

        learning_rate: float, default=0.5
            The initial learning rate used.

        lr_scheduler: str in {'cosine_annealing', 'linear_warmup_cosine_annealing', 'constant'} or 
            LRSchedulerPLType or class or None, default='cosine_annealing'
            The learning rate scheduler to use.

        lr_scheduler_kwargs: dictionary or None, default=dict(eta_min=5e-4)
            Additional keyword arguments for the learning rate scheduler.
            By default, it sets the minimum learning rate to 5e-4 for the cosine annealing scheduler.
            If `lr_scheduler` is an instantiated object, these kwargs are ignored.
        
        **kwargs: dict, optional
            Additional keyword arguments for the BaseEstimator class, such as `max_epochs`, `max_steps`,
            `num_sanity_val_steps`, `check_val_every_n_epoch`, `callbacks`, etc.

        Attributes
        ----------
        encoder_: torch.nn.Module
            Deep neural network trained to map input data to low-dimensional vectors.
            
        loss_: RnCLoss
            The RnC loss function used for training the model.
        """
        super().__init__(**kwargs)
        self.encoder = encoder
        self.encoder_kwargs = encoder_kwargs if encoder_kwargs is not None else {}
        self.n_embedding = n_embedding
        self.temperature = temperature
        self.label_diff = label_diff
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
        self: RnC
            The fitted model.

        """
        # Instantiate the encoder + projection head + loss
        self.encoder_ = self._build_encoder(self.encoder, self.encoder_kwargs)
        self.loss_ = self._build_loss(self.temperature)
        self._cache = []

        # Fit the model
        return super().fit(train_dataloader, val_dataloader)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ 
        Forward pass through the encoder to obtain the embeddings.

        Parameters
        ----------
        X: torch.Tensor
            Input data tensor.

        Returns
        -------
        embeddings: torch.Tensor
            The output embeddings.
        """
        X = X.to(self.device)
        Z = self.encoder_(X)
        return Z
    

    def training_step(self,  batch: Any, batch_idx: int):
        """ Perform one training step during an epoch and computes training loss.

        Parameters
        ----------
        batch: Any 
            A batch of data that has been generated from train_dataloader.
            It should be a triplet of Tensors (V1, V2, y) where V1 and V2 
            are two views of the same sample and y is the (continuous) label. 
        
        batch_idx: int
            The index of the current batch.
        
        Returns
        ----------
        loss: Tensor
            Training loss computed on this batch of data.
        """
        V1, V2, y = self.parse_batch(batch)
        Z1, Z2 = self.encoder_(V1), self.encoder_(V2)
        loss = self.loss_(Z1, Z2, y)
        self.log("loss/train", loss, prog_bar=True)
        outputs = { 
            "loss": loss, 
            "Z1": Z1.cpu().detach(), 
            "Z2": Z2.cpu().detach(), 
            "y_true": y.cpu().detach() if y is not None else None
        }
        # Returns everything needed for further logging/metrics computation
        return outputs    

    def validation_step(self,  batch: Any, batch_idx: int):
        """ Only computes the validation embedding for further metrics computation.

        Parameters
        ----------
        batch: Any
            A batch of data that has been generated from val_dataloader.
            It should be a pair of Tensors (V1, V2) or a tuple (V1, V2, y) 
            where V1 and V2 are two views of the same sample. 
            If (V1, V2, y) is given, y is ignored.
        
        batch_idx: int
            The index of the current batch.
        """
    
        V1, V2, y = self.parse_batch(batch)
        Z1, Z2 = self.encoder_(V1), self.encoder_(V2)
        outputs = {
            "Z1": Z1.cpu().detach(), 
            "Z2": Z2.cpu().detach(), 
            "y_true": y.cpu().detach() if y is not None else None
        }
        self._cache.append(outputs)
        # Returns everything needed for further logging/metrics computation
        return outputs 


    def parse_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Parses the batch to return the two views of the data.

        Parameters
        ----------
        batch: Any
            A batch of data that has been generated from train_dataloader or val_dataloader.
            It should be a tuple (V1, V2, y) where V1 and V2 are two views of the same sample
            and y is the (continuous) label.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
            Tuple (V1, V2, y)
        """
        if isinstance(batch, Sequence) and len(batch) == 3:
            return batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
        elif isinstance(batch, torch.Tensor) and len(batch) == 3:
            return batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
        else:
            raise ValueError("batch should be a tuple of 3 " \
            "Tensors (representing two views and label), got %s" % type(batch))
    

    def on_validation_epoch_end(self):
        """ Computes the validation loss across the entire validation set. """

        Z1 = torch.cat([out["Z1"] for out in self._cache], dim=0).to(self.device)
        Z2 = torch.cat([out["Z2"] for out in self._cache], dim=0).to(self.device)
        y = torch.cat([out["y_true"] for out in self._cache], dim=0).to(self.device)
        self.log("loss/val", self.loss_(Z1, Z2, y))
        # Free the cache
        self._cache = []


    def configure_optimizers(self):
        known_optimizers = {
            "adam": torch.optim.Adam,
            "adamW": torch.optim.AdamW,
            "sgd": torch.optim.SGD
        }
        params = list(self.encoder_.parameters()) + list(self.projection_head_.parameters())
        if isinstance(self.optimizer, str):
            if self.optimizer not in known_optimizers:
                raise ValueError(f"Optimizer '{self.optimizer}' is not implemented. "
                                 f"Please use one of the available optimizers: "
                                 f"{', '.join(known_optimizers.keys())}")
            optimizer = known_optimizers[self.optimizer](
                params=params,
                lr=self.learning_rate,
                **self.optimizer_kwargs
            )
        elif isinstance(self.optimizer, Optimizer):
            if len(self.optimizer_kwargs) > 0:
                logging.getLogger(__name__).warning("optimizer is already instantiated, ignoring 'optimizer_kwargs'")
            optimizer = self.optimizer
        elif isinstance(self.optimizer, type) and issubclass(self.optimizer, Optimizer):
            optimizer = self.optimizer(
                params=params,
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
                "linear_warmup_cosine_annealing": LinearWarmupCosineAnnealingLR,
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
            "cebra": Offset0ModelMSE,
            "vit": VisionTransformer
        }

        if "n_embedding" in encoder_kwargs:
            logging.getLogger(__name__).warning("n_embedding is specified in encoder_kwargs, " \
                                "it will override the n_embedding parameter")
            self.n_embedding = encoder_kwargs["n_embedding"]
        else:
            encoder_kwargs["n_embedding"] = self.n_embedding

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


    def _build_loss(self, temperature: float, label_diff: float) -> nn.Module:
        """ Builds the RnC loss function with the specified temperature.

        Parameters
        ----------
        temperature: float
            Scaling parameter in the similarity function between two latent representations.
        
        label_diff: str in {'l1', 'l2'}
            Which distance to use between labels, ultimately used to rank 
            the samples according to this distance matrix.

        Returns
        -------
        loss: nn.Module
            The RnC loss function.
        """
        return RnCLoss(temperature=temperature, label_diff=label_diff)
