from pytorch_lightning.utilities.types import LRSchedulerPLType
from torch.optim import Optimizer
import torch.nn as nn
from typing import  Dict, Any, Union, Optional, Type, Sequence, Tuple, List
import torch
import logging
import copy

# This library
from nidl.models import BaseEstimator, EmbeddingTransformerMixin
from nidl.volume.backbones import vit_tiny, vit_small, vit_base, vit_large
from nidl.volume.utils.jepa_predictor import VisionTransformerPredictor
from nidl.utils.lr_schedulers import LinearWarmupCosineAnnealingLR


class JEPA(BaseEstimator, EmbeddingTransformerMixin):
    """Implementation of I-JEPA [1]
        
    Solver that predicts the representations of missing parts of an image based on its surrounding context. It uses
    two encoders (context encoder and target encoder) to obtain the contextual and target features and a third predictor
    network to obtain the predictions. 
        
    The target encoder is an Exponential Moving Average (EMA) of the context encoder. Only the target encoder is used 
    for evaluation (it is considered as the "encoder" in our case since the context encoder and predictor are dropped).


    [1] Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture, Assran et al., CVPR 2023
    """

    def __init__(self, 
                 encoder: Union[str, nn.Module, Type[nn.Module]]="vit_tiny_3d", 
                 encoder_kwargs: Optional[Dict[str, Any]]={"pool": None},
                 predictor: Union[str, nn.Module, Type[nn.Module]]="lightweight_vit_3d",
                 predictor_kwargs: Optional[Dict[str, Any]]=None,
                 ema_start: float=0.996,
                 ema_end: float=1.0,
                 optimizer: Union[str, Optimizer, Type[Optimizer]]="adamW", 
                 optimizer_kwargs: Optional[Dict[str, Any]]=dict(betas=(0.9, 0.99), weight_decay=5e-5),
                 learning_rate: float=1e-3,
                 lr_scheduler: Optional[Union[
                     str, LRSchedulerPLType, Type[LRSchedulerPLType]]]="linear_warmup_cosine_annealing",
                 lr_scheduler_kwargs: Optional[Dict[str, Any]]=dict(
                     warmup_epochs=40, warmup_start_lr=2e-4, eta_min=1e-6),
                 **kwargs: Any
                ):
        """
        Parameters
        ----------
        encoder: str in {'vit_tiny_3d', 'vit_small_3d', 'vit_base_3d', 'vit_large_3d'} 
            or nn.Module or class, default='vit_tiny_3d'
            Which encoder backbone to use for embedding the input. 
            It corresponds to the target encoder in I-JEPA. The same architecture is 
            used for the context encoder during training. This encoder must take as 
            input images and eventual masks.
            If not a string, a PyTorch :class:`~torch.nn.Module` is expected. 
            In general, the uninstantiated class should be passed, although instantiated
            modules will also work.  

        encoder_kwargs: dictionary or None, default={"pool": None}
            It specifies the options for building the encoder (depends on each architecture).
            If `n_embedding` is specified here, it will override the `n_embedding` parameter.
        
        predictor: str in {'lightweight_vit_3d'} or nn.Module or class, default='lightweight_vit_3d'
            Predictor model (usually lightweight compared to the context/target encoders) trained to predict
            the masked part of an image from a context in the latent space. 
            If not a string, a PyTorch :class:`~torch.nn.Module` is expected. 
            In general, the uninstantiated class should be passed, although instantiated
            modules will also work.

        predictor_kwargs: dictionary or None, default=None
            It specifies the options for building the predictor (depends on each architecture).
        
        ema_start: float, default=0.996
            Exponential Moving Average momentum starting value. 
            It varies linearly between 'ema_start' and 'ema_stop' during training.
        
        ema_end: float, default=1.0
            Exponential Moving Average momentum ending value. 

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
            It corresponds to the target encoder in the JEPA model.
            NB: the context encoder is not saved after training. 
        
        predictor_: torch.nn.Module
           Predictor model trained to predict the masked part of an image 
           from a context in the latent space. This can be useful at inference time
           to predict missing part of the input. 
    
        loss_: SmoothL1Loss
            Smooth l1 loss used for training the model.
        """
        super().__init__(**kwargs)
        self.encoder = encoder
        self.encoder_kwargs = encoder_kwargs if encoder_kwargs is not None else {}
        self.predictor = predictor
        self.predictor_kwargs = predictor_kwargs if predictor_kwargs is not None else {}
        self.ema_start = ema_start
        self.ema_end = ema_end
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
        self: JEPA
            The fitted model.

        """
        # Instantiate the (target) encoder + context encoder + predictor + loss
        self.encoder_ = self._build_encoder(self.encoder, self.encoder_kwargs)
        self.context_encoder = copy.deepcopy(self.encoder_) # only for training
        self.predictor_ = self._build_predictor(self.predictor, self.predictor_kwargs)
        self.loss_ = self._build_loss()
        self._cache = []

        # Fit the model
        return super().fit(train_dataloader, val_dataloader)

    def on_train_start(self):
        # Compute momentum scheduler
        num_iter = None
        if self.trainer.max_epochs is not None:
            num_iter = int(len(self.trainer.train_dataloader)*self.trainer.max_epochs)
        if self.trainer.max_steps is not None:
            if num_iter is not None:
                num_iter = min(self.trainer.max_steps, num_iter)
            else:
                num_iter = self.trainer.max_steps
        self.momentum_scheduler = (self.ema_start + i*(self.ema_end-self.ema_start)/num_iter
                          for i in range(num_iter+1))


    def training_step(self,  batch: Any, batch_idx: int):
        """ Perform one training step during an epoch and computes training loss.

        Parameters
        ----------
        batch: Any
            A batch of data generated from train_dataloader, usually through the `nidl.data.collate.MaskCollateFunction` 
            collate function. It should be a triplet `(X, masks_enc, masks_pred)` or quadruplet `(X, y, masks_enc, masks_pred)`
            where `X` is a torch.Tensor representing images, `masks_enc` and `masks_pred` are lists of 1D torch.Tensor 
            (representing indices). If a quadruplet `(X, y, masks_enc, masks_pred)` is given, the labels `y` are ignored.
        
        batch_idx: int
            The index of the current batch.
        
        Returns
        ----------
        loss: Tensor
            Training loss computed on this batch of data.
        """
    
        X, masks_enc, masks_pred = self.parse_batch(batch)

        # Update target encoder with moving average
        JEPA.update_momentum(self.context_encoder, self.encoder_, m=next(self.momentum_scheduler))

        # Compute target features
        with torch.no_grad():
            h = self.encoder_(X)
            h = nn.functional.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
            B = len(h)
            # create targets (masked regions of h)
            h = JEPA.apply_masks(h, masks_pred)
            # repeats targets for each context (usually len(masks_enc)=1)
            h = JEPA.repeat_interleave_batch(h, B, repeat=len(masks_enc))
        print("ok 3", flush=True)

        # Compute context features
        z = self.context_encoder(X, masks_enc)
        z = self.predictor_(z, masks_enc, masks_pred)

        # Forward context 
        loss = self.loss_(z, h)
        self.log("loss/train", loss, prog_bar=True)
        # Returns everything needed for further logging/metrics computation
        return {
            "loss": loss,
            "context": z.cpu().detach(),
            "target": h.cpu().detach()
        }


    def validation_step(self,  batch: Any, batch_idx: int):
        """ Only computes the validation embedding for further metrics computation.

        Parameters
        ----------
        batch: Any
            A batch of data generated from train_dataloader, usually through the `nidl.data.collate.MaskCollateFunction` 
            collate function. It should be a triplet `(X, masks_enc, masks_pred)` or quadruplet `(X, y, masks_enc, masks_pred)`
            where `X` is a torch.Tensor representing images, `masks_enc` and `masks_pred` are lists of 1D torch.Tensor 
            (representing indices). If a quadruplet `(X, y, masks_enc, masks_pred)` is given, the labels `y` are ignored.
        
        batch_idx: int
            The index of the current batch.
        """

        X, masks_enc, masks_pred = self.parse_batch(batch)

        h = self.encoder_(X)
        h = nn.functional.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
        B = len(h)
        # create targets (masked regions of h)
        h = JEPA.apply_masks(h, masks_pred)
        # repeats targets for each context (usually len(masks_enc)=1)
        h = JEPA.repeat_interleave_batch(h, B, repeat=len(masks_enc))
        
        # Compute context features
        z = self.context_encoder(X, masks_enc)
        z = self.predictor_(z, masks_enc, masks_pred)

        # Compute validation loss
        val_loss = self.loss_(z, h)
        self.log("loss/val", val_loss, prog_bar=True)

        # Returns everything needed for further logging/metrics computation
        return {
            "loss": val_loss,
            "context": z.cpu().detach(),
            "target": h.cpu().detach()
        }


    @staticmethod
    def apply_masks(x, masks):
        """
        Parameters
        ----------
        x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
            Input data from which to extract the patches.
        
        masks: list of tensors
            List of M tensors containing indices of patches in [N] *to keep*. 
            Each mask has shape [B, K] where K is the number of patches to keep.

        Returns
        ----------
        torch.Tensor: tensor of shape [B * M, K, D]
            Images with only visible parts extracted as defined by the input masks.
        """
        all_x = []
        for m in masks:
            # Expand mask to shape [B, K, D]
            mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
            # Select the K patches from each element in the batch 
            all_x += [torch.gather(x, dim=1, index=mask_keep)]
        return torch.cat(all_x, dim=0)
    

    @staticmethod
    def repeat_interleave_batch(x, B, repeat):
        """ Repeat each batch of size B 'repeat' times and preserves the original batch orderering. 
        
        Example
        ----------
        >>> x = torch.arange(6)  # [0, 1, 2, ..., 5]
        >>> B = 2
        >>> repeat = 3
        >>> repeat_interleave_batch(x, B, repeat) == \
            [0,1, 0,1, 0,1, 2,3, 2,3, 2,3, 4,5, 4,5, 4,5]

        """
        N = len(x) // B
        x = torch.cat([
            torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
            for i in range(N)
        ], dim=0)
        return x
    
    @staticmethod
    @torch.no_grad()
    def update_momentum(model: nn.Module, model_ema: nn.Module, m: float):
        """Updates parameters of `model_ema` with Exponential Moving Average of `model`

        Momentum encoders are a crucial component for models such as JEPA.

        Args:
            model:
                The current model.
            model_ema:
                The model with exponential moving average (EMA) parameters.
            m:
                The momentum factor, between 0 and 1.

        Examples:
            >>> backbone = resnet18()
            >>> projection_head = MoCoProjectionHead()
            >>> backbone_momentum = copy.deepcopy(moco)
            >>> projection_head_momentum = copy.deepcopy(projection_head)
            >>>
            >>> # update momentum
            >>> update_momentum(moco, moco_momentum, m=0.999)
            >>> update_momentum(projection_head, projection_head_momentum, m=0.999)
        """
        for model_ema, model in zip(model_ema.parameters(), model.parameters()):
            model_ema.data = model_ema.data * m + model.data * (1.0 - m)


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
    

    def parse_batch(self, batch: Any):
        """ Parses the batch to return input data, eventual labels, context masks and target masks.
        TODO: add more check on the data (shape, type). 

        Parameters
        ----------
        batch: Any
            A batch of data generated from train_dataloader, usually through the `nidl.data.collate.MaskCollateFunction` 
            collate function. It should be a triplet `(X, masks_enc, masks_pred)` or quadruplet `(X, y, masks_enc, masks_pred)`
            where `X` is a torch.Tensor representing images, `masks_enc` and `masks_pred` are lists of 1D torch.Tensor 
            (representing indices). If a quadruplet `(X, y, masks_enc, masks_pred)` is given, the labels `y` are ignored.


        Returns
        -------
        tuple of (torch.Tensor, List[torch.Tensor], List[torch.Tensor])
        """
        X, y, masks_enc, masks_pred = None, None, None, None

        if isinstance(batch, Sequence) and len(batch) == 3:
            X, masks_enc, masks_pred = batch[0], batch[1], batch[2]
        elif isinstance(batch, Sequence) and len(batch) == 4:
            X, y, masks_enc, masks_pred = batch
        else:
            raise ValueError("batch should be a tuple of " \
                             "3 or 4 Tensors, got %s" % type(batch))
        
        assert isinstance(masks_enc, list) and isinstance(masks_pred, list)
        assert isinstance(X, torch.Tensor)
        X = X.to(self.device)
        masks_enc = [m.to(self.device) for m in masks_enc]
        masks_pred = [m.to(self.device) for m in masks_pred]
        return X, masks_enc, masks_pred
        

    def configure_optimizers(self):
        # TODO: handle weight decay differently between groups of parameters
        known_optimizers = {
            "adam": torch.optim.Adam,
            "adamW": torch.optim.AdamW,
            "sgd": torch.optim.SGD
        }
        params = list(self.context_encoder.parameters()) + list(self.predictor_.parameters())
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
            "vit_tiny_3d": vit_tiny,
            "vit_small_3d": vit_small,
            "vit_base_3d": vit_base,
            "vit_large": vit_large
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
    

    def _build_predictor(self,
                         predictor:Union[str, nn.Module, Type[nn.Module]], 
                         predictor_kwargs: Dict[str, Any]) -> nn.Module:
        known_predictors = {
            "lightweight_vit_3d": VisionTransformerPredictor
        }
        if isinstance(predictor, str):
            if predictor not in known_predictors:
                raise ValueError(f"Predictor '{predictor}' is not implemented. "
                                 f"Please use one of the available encoders: "
                                 f"{', '.join(known_predictors.keys())}")
            predictor = known_predictors[predictor](**predictor_kwargs)
        elif isinstance(predictor, nn.Module):
            if predictor_kwargs is not None and len(predictor_kwargs) > 0:
                logging.getLogger(__name__).warning("predictor_kwargs is already instantiated, ignoring 'predictor_kwargs'")
        elif isinstance(predictor, type) and issubclass(predictor, nn.Module):
            predictor = predictor(**predictor_kwargs)
        else:
            raise ValueError(f"Predicto must be a string, a PyTorch nn.Module, or a class "
                                      f"inheriting from nn.Module, got {type(predictor)}")
        return predictor


    def _build_loss(self) -> nn.Module:
        """ Builds the smooth l1 loss function.

        Returns
        -------
        loss: nn.Module
            The InfoNCE loss function.
        """
        return nn.SmoothL1Loss()
