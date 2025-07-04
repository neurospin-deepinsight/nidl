from typing import  Dict, Any, Union, Optional, Type, Sequence, Tuple, List
import logging
from pytorch_lightning.utilities.types import LRSchedulerPLType
from torch.optim import Optimizer
from torchvision.ops import MLP
import torch.nn as nn
import torch

from ..base import BaseEstimator, TransformerMixin
from ...volume.backbones import AlexNet, resnet18, resnet50, densenet121
from ...losses import yAwareInfoNCE
from .utils.projection_heads import yAwareProjectionHead


class yAwareContrastiveLearning(TransformerMixin, BaseEstimator):
    """y-Aware Contrastive Learning implementation [1]

    y-Aware Contrastive Learning is a self-supervised learning framework for learning visual representations with 
    auxiliary variables. It leverages contrastive learning by maximizing the agreement between 
    differently augmented views of images with similar auxiliary variables while minimizing 
    agreement between different images. The framework consists of:
    
    1) Data Augmentation – Generates two augmented views of an image.
    2) Kernel - Similarity function between auxiliary variables. 
    2) Encoder (Backbone Network) – Maps images to feature embeddings (e.g., 3D-ResNet).
    3) Projection Head – Maps features to a latent space for contrastive loss optimization.
    4) Contrastive Loss (y-Aware) – Encourages augmented views of i) the same image and ii) images with close auxiliary variables
       to be closer while pushing dissimilar ones apart.

    [1] Contrastive Learning with Continuous Proxy Meta-Data for 3D MRI Classification, Dufumier et al., MICCAI 2021

    Examples
    --------
    >>> from nidl.estimators.ssl import yAwareContrastiveLearning
    >>> from nidl.datasets.openbhb import OpenBHB
    >>> from nidl.volume.transforms import yAwareTransformStrong
    >>> from torch.utils.data import DataLoader
    >>> yaware = yAwareContrastiveLearning(encoder="resnet18_3d", projection_head_kwargs={"input_dim": 512})    
    >>> openBHB_train = OpenBHB(".", train=True)
    >>> openBHB_test = OpenBHB(".", train=False)
    >>> collate_fn = TwoViewsCollateFunction(yAwareTransformStrong())  
    >>> train_dataloader = DataLoader(openBHB_train, batch_size=32, shuffle=True)
    >>> test_dataloader = DataLoader(openBHB_test, batch_size=32, shuffle=False)
    >>> yaware.fit(train_dataloader)
    >>> test_embedding = yaware.transform(test_dataloader)

    """

    def __init__(self, 
                 encoder: Union[str, nn.Module, Type[nn.Module]]="alexnet_3d", 
                 encoder_kwargs: Optional[Dict[str, Any]]=None,
                 projection_head: Union[None, str, nn.Module, Type[nn.Module]]="yaware",
                 projection_head_kwargs: Optional[Dict[str, Any]]={"input_dim": 128},
                 temperature: float=0.1,
                 kernel: str="gaussian",
                 bandwidth: Union[str, int, float, List[float]]="scott",
                 optimizer: Union[str, Optimizer, Type[Optimizer]]="adam", 
                 optimizer_kwargs: Optional[Dict[str, Any]]=dict(betas=(0.9, 0.99), weight_decay=5e-5),
                 learning_rate: float=1e-4,
                 lr_scheduler: Optional[Union[LRSchedulerPLType, Type[LRSchedulerPLType]]]=None,
                 lr_scheduler_kwargs: Optional[Dict[str, Any]]=None,
                 **kwargs: Any
                ):
        """
        Parameters
        ----------
        encoder: str in {'alexnet_3d', 'resnet18_3d', 'resnet50_3d', 'densenet121_3d', 'mlp'} 
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
        
        projection_head: str in {'yaware'} or nn.Module or class or None, default='yaware'
            Which projection head to use for the model. 
            If None, no projection head is used and the the encoder output is directly used for loss computation.
            If not in the default heads, a PyTorch :class:`~torch.nn.Module` is expected. 
            In general, the uninstantiated class should be passed, although instantiated
            modules will also work.
            By default, a 2-layer MLP with ReLU activation, 512-d hidden units and 
            128-d output dimensions is used.

        projection_head_kwargs: dictionary or None, default=None
            It specifies the options for building the projection head (depends on each architecture).
        
        n_embedding: int, default=16
            Dimension of the embedding space.

        temperature: float, default=0.1
            Small values implies more uniformity between samples' embedding whereas high values
            imposes clustered embedding more sensitive to augmentations.
        
        kernel: {'gaussian', 'epanechnikov', 'exponential', 'linear', 'cosine'}, default="gaussian"
            Kernel used as similarity function between auxiliary variables. 

        bandwidth: str in {'scott', 'silverman'} or float or list of float, default="scott"
            The method used to calculate the bandwidth in kernel:
            * If `bandwidth` is str, must be scott's or 'silverman's for automatic 
                bandwidth estimation (see Rosenblatt, M. (1956) or Parzen, E. (1962))
            * If `bandwidth` is float, it sets the bandwidth to H=diag(scalar)
            * If `bandwidth` is a list, it sets the bandwidth to H=diag(list) and it
                must be of length equal to the number of features in the auxiliary variables 'y'.

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

        lr_scheduler: LRSchedulerPLType or class or None, default=None
            The learning rate scheduler to use.
        
        lr_scheduler_kwargs: dictionary or None, default=None
            Additional keyword arguments for the learning rate scheduler.
            If `lr_scheduler` is an instantiated object, these kwargs are ignored.
        
        **kwargs: dict, optional
            Additional keyword arguments for the BaseEstimator class, such as `max_epochs`, `max_steps`,
            `num_sanity_val_steps`, `check_val_every_n_epoch`, `callbacks`, etc.

        Attributes
        ----------
        encoder: torch.nn.Module
            Deep neural network trained to map input data to low-dimensional vectors.
        
        projection_head: torch.nn.Module
            Projection head that maps the output of the encoder to a latent space for contrastive loss optimization.
    
        loss: yAwareInfoNCE
            The yAwareInfoNCE loss function used for training the model.
        """
        super().__init__(**kwargs)
        self.encoder_kwargs = encoder_kwargs if encoder_kwargs is not None else {}
        self.encoder = self._build_encoder(encoder, self.encoder_kwargs)
        self.projection_head_kwargs = projection_head_kwargs if projection_head_kwargs is not None else {}
        self.projection_head = self._build_projection_head(projection_head)
        self.temperature = temperature
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.loss = self._build_loss(self.temperature, self.kernel, self.bandwidth)
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs if lr_scheduler_kwargs is not None else {}
        self.optimizer_kwargs = optimizer_kwargs
    

    def training_step(self,  batch: Any, batch_idx: int):
        """ Perform one training step during an epoch and computes training loss.

        Parameters
        ----------
        batch: Any 
            A batch of data that has been generated from train_dataloader.
            It can be a pair (V1, V2) or a triplet of torch.Tensor (V1, V2, y) where 
            V1 and V2 are the two views of the same sample and y is the auxiliary variable.
        
        batch_idx: int
            The index of the current batch.
        
        Returns
        ----------
        loss: Tensor
            Training loss computed on this batch of data.
        """
        V1, V2, y = self.parse_batch(batch)
        Z1, Z2 = self.projection_head(self.encoder(V1)), self.projection_head(self.encoder(V2))
        loss = self.loss(Z1, Z2, y)
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
            A batch of data that has been generated from val_dataloader (if not None).
            It can be a pair (V1, V2) or a triplet of torch.Tensor (V1, V2, y) where 
            V1 and V2 are the two views of the same sample and y is the auxiliary variable.
        
        batch_idx: int
            The index of the current batch.
        """
    
        V1, V2, y = self.parse_batch(batch)
        Z1, Z2 = self.projection_head(self.encoder(V1)), self.projection_head(self.encoder(V2))
        val_loss = self.loss(Z1, Z2, y)
        outputs = {
            "loss": val_loss,
            "Z1": Z1.cpu().detach(), 
            "Z2": Z2.cpu().detach(), 
            "y_true": y.cpu().detach() if y is not None else None
        }
        self.log("loss/val", val_loss)
        # Returns everything needed for further logging/metrics computation
        return outputs 


    def parse_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """ Parses the batch to extract the two views and the auxiliary variable.

        Parameters
        ----------
        batch: Any
            A batch of data that has been generated from train_dataloader or val_dataloader.
            It can be a pair (V1, V2) or a triplet of torch.Tensor (V1, V2, y) where V1 and V2 
            are two views of the same samples and y is the auxiliary variable.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor or None]
            The two views and the auxiliary variable.
        """
        if isinstance(batch, Sequence) and len(batch) == 2:
            V1, V2, y = batch[0].to(self.device), batch[1].to(self.device), None
        elif isinstance(batch, Sequence) and len(batch) == 3:
            V1, V2, y = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
        elif isinstance(batch, torch.Tensor) and len(batch) == 2:
            V1, V2, y = batch[0].to(self.device), batch[1].to(self.device), None
        elif isinstance(batch, torch.Tensor) and len(batch) == 3:
            V1, V2, y = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
        else:
            raise ValueError("batch should be a tuple of two Tensors (representing two views), or a triplet " \
                             "of Tensors (representing two views and an auxiliary variable), got %s"%type(batch))
        return V1, V2, y
    

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
        elif isinstance(self.lr_scheduler, LRSchedulerPLType):
            if len(self.lr_scheduler_kwargs) > 0:
                logging.getLogger(__name__).warning("lr_scheduler is already instantiated, ignoring 'lr_scheduler_kwargs'")
            scheduler = self.lr_scheduler
        elif isinstance(self.lr_scheduler, type) and issubclass(self.lr_scheduler, LRSchedulerPLType):
            scheduler = self.lr_scheduler(
                optimizer=optimizer,
                **self.lr_scheduler_kwargs
            )
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
            "mlp": MLP
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


    def _build_projection_head(self,
                               projection_head: Union[str, nn.Module, Type[nn.Module]], 
                               projection_head_kwargs: Dict[str, Any]) -> nn.Module:
        known_heads = {
            "yaware": yAwareProjectionHead
        }

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
    

    def _build_loss(self, 
                    temperature: float, 
                    kernel: str,
                    bandwidth: Union[str, int, float, List[float]]) -> nn.Module:
        """ Builds the InfoNCE loss function with the specified temperature.

        Parameters
        ----------
        temperature: float
            The temperature parameter for the InfoNCE loss.
        
        kernel: {'gaussian', 'epanechnikov', 'exponential', 'linear', 'cosine'}
            Kernel used as similarity function between auxiliary variables. 

        bandwidth: str in {'scott', 'silverman'} or float or list of float
            The method used to calculate the bandwidth in kernel.

        Returns
        -------
        loss: nn.Module
            The y-Aware InfoNCE loss function.
        """
        return yAwareInfoNCE(kernel=kernel, bandwidth=bandwidth, temperature=temperature)