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
from nidl.models.ssl.utils.heads import VICRegProjectionHead
from nidl.utils.lr_schedulers import LinearWarmupCosineAnnealingLR
from nidl.losses import VICRegLoss


class VICReg(BaseEstimator, EmbeddingTransformerMixin):
    """ VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning [1]

    This code is an adaptation of the original repository: https://github.com/facebookresearch/vicreg to deal
    with neuroimaging data (3D volumetric data with specific data augmentation based mainly on crop).

    VICReg (Variance-Invariance-Covariance Regularization) is a method that maximize agreement between different
    views of the same data (as in SimCLR) but without contrastive constraints. Rather, it avoids the collapse 
    problem by imposing two constraints:
    1. Maximize variance of each features above a given threshold
    2. Minimize covariance between features in the latent space (redundancy reduction, akin to Barlow Twins)
    
    The framework consists of:
    
    1) Data Augmentation – Generates two augmented views of an image (same as SimCLR).
    2) Encoder (Backbone Network) – Maps images to feature embeddings (e.g., 3D-ResNet).
    3) Projection Head – Maps features to a latent space for VicReg loss optimization 
        (usually much larger than representation space!)
    4) VicReg Loss – Encourages similar images to be closer while avoiding redundancy in the features.

    This nidl estimator has `fit`, `transform` and `fit_transform` methods implemented.

    [1] VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning, Bardes et al., ICLR 2022

    Examples
    --------
    >>> from nidl.models.ssl import VICReg
    >>> vicreg = VICReg(encoder="resnet18_3d", n_embedding=8)

    Example 1: OpenBHB
    >>> from nidl.datasets.openbhb import OpenBHB
    >>> from nidl.data.collate import TwoViewsCollateFunction
    >>> from nidl.volume.transforms import SimCLRTransform
    >>> openBHB_train = OpenBHB("/neurospin/hc/openBHB", train=True)
    >>> openBHB_test = OpenBHB("/neurospin/hc/openBHB", train=False)
    >>> collate_fn = TwoViewsCollateFunction(SimCLRTransform())  
    >>> train_dataloader = DataLoader(openBHB_train, collate_fn=collate_fn, batch_size=32, shuffle=True)
    >>> test_dataloader = DataLoader(openBHB_test, batch_size=32, shuffle=False)
    >>> vicreg.fit(train_dataloader)
    >>> test_embedding = vicreg.transform(test_dataloader)
    """

    def __init__(self, 
                 encoder: Union[str, nn.Module, Type[nn.Module]]="alexnet_3d", 
                 encoder_kwargs: Optional[Dict[str, Any]]=None,
                 projection_head: Union[None, str, nn.Module, Type[nn.Module]]="vicreg",
                 projection_head_kwargs: Optional[Dict[str, Any]]=None,
                 n_embedding: int=16,
                 sim_coeff: float=25.0, 
                 std_coeff: float=25.0, 
                 cov_coeff: float=1.0,
                 optimizer: Union[str, Optimizer, Type[Optimizer]]="adamW", 
                 optimizer_kwargs: Optional[Dict[str, Any]]=dict(betas=(0.9, 0.99), weight_decay=5e-5),
                 learning_rate: float=1e-4,
                 lr_scheduler: Optional[Union[str, LRSchedulerPLType, Type[LRSchedulerPLType]]]="linear_warmup_cosine_annealing",
                 lr_scheduler_kwargs: Optional[Dict[str, Any]]=dict(warmup_epochs=10, warmup_start_lr=1e-6),
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
        
        projection_head: str in {'vicreg'} or nn.Module or class or None, default='vicreg'
            Which projection head to use for the model. 
            If None, no projection head is used and the encoder output is directly used for loss computation.
            If not in the default heads, a PyTorch :class:`~torch.nn.Module` is expected. 
            In general, the uninstantiated class should be passed, although instantiated
            modules will also work.
            By default, a 2-layer MLP with ReLU activation, 2048-d hidden units and 
            128-d output dimensions is used.

        projection_head_kwargs: dictionary or None, default=None
            It specifies the options for building the projection head (depends on each architecture).
        
        n_embedding: int, default=16
            Dimension of the embedding space.

        sim_coeff: float, default=25.0
            Invariance regularization loss coefficient.
        
        std_coeff: float, default=25.0
            Variance regularization loss coefficient.

        cov_coeff: float, default=1.0
            Covariance regularization loss coefficient.

        optimizer: str in {'sgd', 'adam', 'adamW'} or Optimizer or class, default='adamW'
            The optimizer to use for training the model. It can be a string:
            'sgd': stochastic gradient descent (with eventually momentum)
            'adam': stochastic first order gradient-based optimizer
            'adamW' (default): Adam optimizer with decoupled weight decay regularization 
               (see ''Decoupled Weight Decay Regularization, Loshchilov and Hutter, ICLR 2019'')
            or an instantiated object of type `torch.optim.Optimizer` or a class inheriting from it.
        
        optimizer_kwargs: dictionary or None, default={'beta_1': 0.9, 'beta_2': 0.99, 'weight_decay': 5e-5}
            Arguments to give to optimizer ('adam' by default) where 'beta_1' and 'beta_2' are the
            exponential decay rate for estimates of first and second moment vector in Adam.
            'weight_decay' is scaling factor for l2 penalization on model's weights. It can be used for
            'adam', 'sgd' and 'adamW' optimizers.
            This is ignored if `optimizer` is an instantiated object.

        learning_rate: float, default=1e-4
            The initial learning rate used.

        lr_scheduler: str in {'linear_warmup_cosine_annealing', 'constant'} or 
            LRSchedulerPLType or class or None, default='linear_warmup_cosine_annealing'
            The learning rate scheduler to use.
        
        lr_scheduler_kwargs: dictionary or None, default=dict(warmup_epochs=10, start_warmup_value=1e-6)
            Additional keyword arguments for the learning rate scheduler.
            If `lr_scheduler` is an instantiated object, these kwargs are ignored.
        
        **kwargs: dict, optional
            Additional keyword arguments for the BaseEstimator class, such as `max_epochs`, `max_steps`,
            `num_sanity_val_steps`, `check_val_every_n_epoch`, `callbacks`, etc.

        Attributes
        ----------
        encoder_: torch.nn.Module
            Deep neural network trained to map input data to low-dimensional vectors.
        
        projection_head_: torch.nn.Module
            Projection head that maps the output of the encoder to a latent space for contrastive loss optimization.
    
        loss_: VICRegLoss
            The VICReg loss function used for training the model.
        """
        super().__init__(**kwargs)
        self.encoder = encoder
        self.encoder_kwargs = encoder_kwargs if encoder_kwargs is not None else {}
        self.projection_head = projection_head
        self.projection_head_kwargs = projection_head_kwargs if projection_head_kwargs is not None else {}
        self.n_embedding = n_embedding
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
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
        self: VICReg
            The fitted model.

        """
        # Instantiate the encoder + projection head + loss
        self.encoder_ = self._build_encoder(self.encoder, self.encoder_kwargs)
        self.projection_head_ = self._build_projection_head(self.projection_head, self.projection_head_kwargs)
        self.loss_ = self._build_loss(self.sim_coeff, self.std_coeff, self.cov_coeff)
        self._cache = dict(val_embedding=[])

        # Fit the model
        return super().fit(train_dataloader, val_dataloader)


    def forward(self, X: torch.Tensor, apply_projection: bool=False) -> torch.Tensor:
        """ 
        Forward pass through the encoder to obtain the embeddings.

        Parameters
        ----------
        X: torch.Tensor
            Input data tensor.
        
        apply_projection: bool, default=False
            If True, applies the projection head to the output of the encoder.

        Returns
        -------
        embeddings: torch.Tensor
            The output embeddings.
        """
        X = X.to(self.device)
        Z = self.encoder_(X)
        if apply_projection:
            Z = self.projection_head_(Z)
        return Z
    

    def training_step(self,  batch: Any, batch_idx: int):
        """ Perform one training step during an epoch and computes training loss.

        Parameters
        ----------
        batch: Any 
            A batch of data that has been generated from train_dataloader.
            It should be a pair of Tensors (V1, V2) or a tuple (V1, V2, y) 
            where V1 and V2 are two views of the same sample. 
            If (V1, V2, y) is given, y is ignored.
        
        batch_idx: int
            The index of the current batch.
        
        Returns
        ----------
        loss: Tensor
            Training loss computed on this batch of data.
        """
        V1, V2 = self.parse_batch(batch)
        Z1, Z2 = self.projection_head_(self.encoder_(V1)), self.projection_head_(self.encoder_(V2))
        loss = self.loss_(Z1, Z2)
        self.log("loss/train", loss, prog_bar=True)
        return loss
    

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
    
        V1, V2 = self.parse_batch(batch)
        Z1, Z2 = self.projection_head_(self.encoder_(V1)), self.projection_head_(self.encoder_(V2))
        self._cache["val_embedding"].extend(np.stack((Z1.cpu().detach().numpy(), 
                                                      Z2.cpu().detach().numpy()), axis=1))


    def parse_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Parses the batch to return the two views of the data.

        Parameters
        ----------
        batch: Any
            A batch of data that has been generated from train_dataloader or val_dataloader.
            It should be a pair of torch.Tensor (V1, V2) or a tuple (V1, V2, y) 
            where V1 and V2 are two views of the same sample. 
            If (V1, V2, y) is given, y is ignored.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            The two views of the data.
        """
        if isinstance(batch, Sequence) and len(batch) == 2:
            return batch[0].to(self.device), batch[1].to(self.device)
        elif isinstance(batch, Sequence) and len(batch) == 3:
            # If the batch is a tuple of (V1, V2, y), we ignore y
            return batch[0].to(self.device), batch[1].to(self.device)
        elif isinstance(batch, torch.Tensor) and len(batch) == 2:
            return batch[0].to(self.device), batch[1].to(self.device)
        else:
            raise ValueError("batch should be a tuple of " \
                             "two Tensors (representing two views), got %s" % type(batch))
    

    def on_validation_epoch_end(self):
        """ Computes the validation loss across the entire validation set. """

        Z = torch.tensor(np.array(self._cache["val_embedding"])).to(self.device)
        Z1, Z2 = Z[:, 0], Z[:, 1]
        self.log("loss/val", self.loss_(Z1, Z2))

        # Free the cache
        self._cache["val_embedding"] = []


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

    def _build_projection_head(self,
                               projection_head: Union[str, nn.Module, Type[nn.Module]], 
                               projection_head_kwargs: Dict[str, Any]) -> nn.Module:
        known_heads = {
            "vicreg": VICRegProjectionHead
        }

        projection_head_kwargs["input_dim"] = self.n_embedding

        if projection_head is None:
            projection_head = nn.Identity()
        elif isinstance(projection_head, str):
            if projection_head not in known_heads:
                raise ValueError(f"Projection head '{projection_head}' is not implemented. "
                                 f"Please use one of the available heads: "
                                 f"{', '.join(known_heads.keys())}")
            projection_head = known_heads[projection_head](**projection_head_kwargs)
        elif isinstance(projection_head, nn.Module):
            if projection_head_kwargs is not None and len(projection_head_kwargs) > 0:
                logging.getLogger(__name__).warning("projection head is already instantiated, ignoring 'projection_head_kwargs'")
        elif isinstance(projection_head, type) and issubclass(projection_head, nn.Module):
            projection_head = projection_head(**projection_head_kwargs)
        else:
            raise ValueError(f"Projection head must be None, a string, a PyTorch nn.Module, or a class "
                             f"inheriting from nn.Module, got {type(projection_head)}")
        return projection_head
    

    def _build_loss(self, sim_coeff: float, std_coeff: float, cov_coeff: float) -> nn.Module:
        """ Builds the VICReg loss function with the specified coefficients.

        Parameters
        ----------
        sim_coeff: float
            The invariance regularization loss coefficient.

        std_coeff: float
            The variance regularization loss coefficient.

        cov_coeff: float
            The covariance regularization loss coefficient.

        Returns
        -------
        loss: nn.Module
            The InfoNCE loss function.
        """
        return VICRegLoss(sim_coeff=sim_coeff, std_coeff=std_coeff, cov_coeff=cov_coeff)
