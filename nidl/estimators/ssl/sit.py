##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Surface vision estimators.
"""

import math
from typing import (
    Any,
)

import torch
from torch import nn
from torch.optim import (
    Optimizer,
)
from torch.optim.lr_scheduler import (
    LRScheduler,
)
from torchmetrics import (
    Metric,
    MetricCollection,
)

from ...surface.backbones import (
    SiTAttentionMaps,
)
from ...utils.instantiate import (
    build_instance,
)
from ..base import (
    BaseEstimator,
    RegressorMixin,
    TransformerMixin,
)


class SiTPretrainer(TransformerMixin, BaseEstimator):
    """
    Surface image Transformers (SiT) pretrainer [1]_ [2]_.

    Pretained a SiT model using Masked Patch Prediction (MPP). MPP
    is a self-supervised learning objective for representation learning.
    The proposed implementation is a self-supervised BERT-style masking
    corruption strategy tailored for cortical data. It randomly corrupts a
    sequence of surface patches using three techniques:

    - replacing with a learnable mask token,
    - swapping with other random patches,
    - or leaving them unchanged.

    The model is then optimized to reconstruct the original
    uncorrupted surface features solely at the corrupted positions using
    Mean Squared Error loss.

    Parameters
    ----------
    model : nn.Module | type[nn.Module]
        The SiT architecture. It can be given as an already instantiated
        Module or as a class of Module (in which case it will be instantiated
        with `model_kwargs`).
    optimizer : Optimizer | type[Optimizer]
        It can be given as an already instantiated Optimizer or as a class of
        Optimizer (in which case it will be instantiated with
        `optimizer_kwargs`).
    lr_scheduler : LRScheduler | type[LRScheduler] | None
        Learning rate scheduler to use. It can be given as an already
        instantiated LRScheduler or as a class of LRScheduler (in
        which case it will be instantiated with `lr_scheduler_kwargs`).
        Default None.
    model_kwargs : dict[str, Any] | None
        Options for building the SiT model. Ignored if `encoder` is
        instantiated.
        Default None.
    optimizer_kwargs : dict[str, Any] | None
        Options for building the SiT optimizer. Ignored if `optimizer` is
        instantiated.
        Default None.
    lr_scheduler_kwargs : dict[str, Any] | None
        Options for building the SiT scheduler. Ignored if `lr_scheduler` is
        instantiated.
        Default None.
    mask_prob : float | tuple[float, float]
        Total ratio of patches selected to undergo corruption routines. If a
        tuple (low, high) is provided, the probability is sampled uniformly
        from [low, high].
        Default 0.75.
    replace_prob : float | tuple[float, float]
        Ratio of corrupted tokens to overwrite with a static mask distribution.
        If a tuple (low, high) is provided, the probability is sampled
        uniformly from [low, high].
        Default 0.8.
    swap_prob : float | tuple[float, float]
        Ratio of corrupted tokens to swap with a foreign sequential location.
        If a tuple (low, high) is provided, the probability is sampled
        uniformly from [low, high].
        Default 0.02.
    random_state: int | None
        Setting a seed for reproducibility.
        Default None.
    **kwargs : Any
        Additional keyword arguments for the BaseEstimator class, such as
        `max_epochs`, `max_steps`, `num_sanity_val_steps`,
        `check_val_every_n_epoch`, `callbacks`, etc.

    Attributes
    ----------
    model : torch.nn.Module
        SiT network mapping input data to low-dimensional vectors.
    optimizer : Optimizer
        Optimizer used for training.
    lr_scheduler : LRScheduler | None
        Learning rate scheduler used for training.
    loss : MSELoss
        The MSELoss loss function used for training.

    References
    ----------
    .. [1] Dahan, Simon et al., Surface Vision Transformers: Attention-Based
    Modelling applied to Cortical Analysis, MIDL, 2022.
    .. [2] Dahan, Simon et al., Surface Vision Transformers: Flexible
    Attention-Based Modelling of Biomedical Surfaces, TMI, 2022.
    """

    def __init__(
            self,
            model: nn.Module | type[nn.Module],
            optimizer: Optimizer | type[Optimizer],
            lr_scheduler: LRScheduler | type[LRScheduler] | None = None,
            model_kwargs: dict[str, Any] | None = None,
            optimizer_kwargs: dict[str, Any] | None = None,
            lr_scheduler_kwargs: dict[str, Any] | None = None,
            mask_prob: float = 0.75,
            replace_prob: float = 0.8,
            swap_prob: float = 0.02,
            random_state: int | None = None,
            **kwargs: Any,
        ) -> None:
        super().__init__(
            random_state=random_state,
            ignore=["model", "callbacks"],
            **(kwargs or {}),
        )

        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.swap_prob = swap_prob

        self.model = build_instance(
            model,
            **(model_kwargs or {}),
        )
        self.optimizer = build_instance(
            optimizer,
            model.parameters,
            **(optimizer_kwargs or {}),
        )
        self.lr_scheduler = build_instance(
            lr_scheduler,
            self.optimizer,
            **(lr_scheduler_kwargs or {}),
        )
        self.loss = nn.MSELoss()

        n_channels, _, n_vertices = self.model.patch_spec
        self.mask_token = nn.Parameter(
            torch.randn(1, 1, n_channels * n_vertices)
        )
        self.to_original = nn.Linear(*self.model.dim_spec)

    def configure_optimizers(
            self,
        ) -> tuple[list[Optimizer], list[LRScheduler]]:
        """
        Initialize the optimizer and learning rate scheduler.

        Returns
        -------
        tuple[list[Optimizer], list[LRScheduler]
            A tuple containing two distinct lists:
            
            * optimizers : list[Optimizer]
                The first list contains the initialized optimizer for the
                network.
            * schedulers : list[LRScheduler]
                The second list contains the learning rate schedulers. Left
                empty here as no learning rate decays are applied.
        """
        if self.lr_scheduler is not None:
            return [self.optimizer], [self.lr_scheduler]
        else:
            return [self.optimizer]

    def apply_masked_patch_prediction(
            self,
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mask patch prediction corruption.

        Parameters
        ----------
        x : torch.Tensor
            Uncorrupted raw surface input batch matching of shape
            `(batch, channels, patches * vertices)`.

        Returns
        -------
        corrupted_x : torch.Tensor
            Rearranged feature tensor sequence matching shape
            `(batch, patches, vertices * channels)`.
        corrupted_mask : torch.Tensor
            Boolean mask tracking indices affected by the corruption
            configuration.
        """
        # Get locations of surface patches to corrupt
        corrupted_mask = self.get_random_patches(x, self.mask_prob)

        # Swapping with other random patches
        corrupted_x = x.clone().detach()
        if self.swap_prob > 0:
            patch_prob = self.prob_mask_like(
                x,
                self.swap_prob / (1 - self.replace_prob),
            ).to(x.device)
            patch_prob = corrupted_mask * patch_prob
            
            random_patches = torch.randint(
                0,
                x.shape[1],
                (x.shape[0], x.shape[1]),
                device=x.device
            )
            randomized_x = corrupted_x[
                torch.arange(corrupted_x.shape[0]).unsqueeze(-1),
                random_patches,
            ]
            corrupted_x[patch_prob] = randomized_x[patch_prob]

        # Replacing with a learnable mask token
        tokens_mask = self.prob_mask_like(
            x,
            self.replace_prob,
        ).to(x.device)
        replace_mask = corrupted_mask & tokens_mask
        corrupted_x[replace_mask] = self.mask_token.to(x.device)

        return corrupted_x, corrupted_mask

    def _shared_step(
            self,
            x: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
        """
        Shared code for training and validation steps.
        """
        # Patch / rearrange
        x = self.model.patchify(x)

        # Corrupt data
        corrupted_x, corrupted_mask = self.apply_masked_patch_prediction(x)

        # Generator output
        encoded_patches = self.model(corrupted_x)
        out_x = self.to_original(encoded_patches)

        # Compute loss
        loss = self.loss(
            out_x[corrupted_mask],
            x[corrupted_mask],
        )

        return {
            "loss": loss,
            "mask": corrupted_mask.detach(),
            "x": x.detach(),
            "corrupted_x": corrupted_x.detach(),
            "encoded_patches": encoded_patches.detach(),
        }

    def training_step(
            self,
            x: torch.Tensor,
            batch_idx: int,
        ) -> dict[str, torch.Tensor]:
        """
        Perform one training step and computes training loss.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, channels, vertices)``.
            It may contain either left- or right-hemisphere data, but must
            first be symmetrized (as in an xhemi-style dataset).
        batch_idx : int
            Position tracking pointer of the current iteration batch.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
            - "loss": the MSE loss computed on this batch;
            - "x": patchify input data of shape
              ``(batch, channels, patches * vertices)``;
            - "corrupted_x": MPP corrupted input patched data;
            - "encoded_patches": all tokens, excluding the class token.
        """
        outputs = self._shared_step(x)
        self.log(
            "loss/train",
            outputs["loss"],
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return outputs

    def validation_step(
            self,
            x: torch.Tensor,
            batch_idx: int,
        ) -> dict[str, torch.Tensor]:
        """
        Perform one validation step and computes validation loss.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, channels, vertices)``.
            It may contain either left- or right-hemisphere data, but must
            first be symmetrized (as in an xhemi-style dataset).
        batch_idx : int
            Position tracking pointer of the current iteration batch.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
            - "loss": the MSE loss computed on this batch;
            - "x": patchify input data of shape
              ``(batch, channels, patches * vertices)``;
            - "corrupted_x": MPP corrupted input patched data;
            - "encoded_patches": all tokens, excluding the class token.
        """
        outputs = self._shared_step(x)
        self.log(
            "loss/val",
            outputs["loss"],
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return outputs

    def transform_step(
            self,
            x: torch.Tensor,
            batch_idx: int,
            dataloader_idx: int = 0,
        ) -> torch.Tensor:
        """
        Encode the input data into the latent space.

        Importantly, we do not apply the projection head here since it is
        not part of the final model at inference time (only used for training).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, channels, vertices)``.
            It may contain either left- or right-hemisphere data, but must
            first be symmetrized (as in an xhemi-style dataset).
        batch_idx: int
            The index of the current batch (ignored).
        dataloader_idx: int
            The index of the dataloader (ignored).
            Default 0.

        Returns
        -------
        torch.Tensor
           All tokens, excluding the class token.
        """
        return self.model(x)

    @classmethod
    def get_random_patches(
            cls,
            patched_texture: torch.Tensor,
            prob: float,
        ) -> torch.Tensor:
        """
        Creates a boolean mask on the token sequence per sample based on a
        masking probability.
    
        This function guarantees a strict, deterministic number of masked
        patches per sample by calculating a fixed budget `max_masked`
        and selecting the top random indices. Each row in the batch will have
        the exact same number of masked values.

        Parameters
        ----------
        patched_texture : torch.Tensor
            Input patched texture of shape
            `(batch, patch, vertices * channels)`.
        prob : float
            The masking probability ratio.

        Returns
        -------
        torch.Tensor
            A patch selection boolean mask of shape `(batch, patch)`.

        Examples
        --------
        >>> inputs = torch.randn(2, 5, 128)
        >>> mask = get_mask_from_prob(inputs, prob=0.4)
        >>> mask.shape
        torch.Size([2, 5])
        >>> mask.sum(dim=-1)
        tensor([2, 2])
        """
        n_batch, n_patch = patched_texture.shape[:2]
        max_masked = math.ceil(prob * n_patch)

        # Generate random values and select the top-k indices to mask
        rand = torch.rand(
            (n_batch, n_patch),
            device=patched_texture.device,
        )
        _, sampled_indices = rand.topk(max_masked, dim=-1)

        # Create the boolean mask directly using scatter
        new_mask = torch.zeros(
            (n_batch, n_patch),
            device=patched_texture.device,
            dtype=torch.bool,
        )
        return new_mask.scatter_(1, sampled_indices, True)

    @classmethod
    def prob_mask_like(
            cls,
            patched_texture: torch.Tensor,
            prob: float,
        ) -> torch.Tensor:
        """
        Generate a random binary mask based on a uniform distribution.

        Each patch position across the batch and sequence length is masked
        independently with a probability defined by `prob`.

        Parameters
        ----------
        patched_texture : torch.Tensor
            Input patched texture of shape
            ``(batch, patch, vertices x channels)``.
        prob : float
            The cutoff probability for uniform selection.

        Returns
        -------
        torch.Tensor
            A boolean mask of shape `(batch, patch)`.

        Examples
        --------
        >>> inputs = torch.randn(2, 4, 128)
        >>> mask = prob_mask_like(inputs, prob=0.5)
        >>> mask.shape
        torch.Size([2, 4])
        """
        n_batch, n_patch = patched_texture.shape[:2]
        return torch.rand(
            (n_batch, n_patch),
            device=patched_texture.device,
        ) < prob


class SiTTrainer(RegressorMixin, BaseEstimator):
    """
    Train a model using a continuous variable.

    Parameters
    ----------
    model : nn.Module | type[nn.Module]
        The SiT architecture. It can be given as an already instantiated
        Module or as a class of Module (in which case it will be instantiated
        with `model_kwargs`).
    optimizer : Optimizer | type[Optimizer]
        It can be given as an already instantiated Optimizer or as a class of
        Optimizer (in which case it will be instantiated with
        `optimizer_kwargs`).
    lr_scheduler : LRScheduler | type[LRScheduler] | None
        Learning rate scheduler to use. It can be given as an already
        instantiated LRScheduler or as a class of LRScheduler (in
        which case it will be instantiated with `lr_scheduler_kwargs`).
        Default None.
    model_kwargs : dict[str, Any] | None
        Options for building the SiT model. Ignored if `encoder` is
        instantiated.
        Default None.
    optimizer_kwargs : dict[str, Any] | None
        Options for building the SiT optimizer. Ignored if `optimizer` is
        instantiated.
        Default None.
    lr_scheduler_kwargs : dict[str, Any] | None
        Options for building the SiT scheduler. Ignored if `lr_scheduler` is
        instantiated.
        Default None.
    metrics : dict[str, Metric]  | None = None
        A dictionary mapping names to torchmetrics metric instances for the
        training/validation phases.
        Default None.
    return_attention : bool
        If True, returns attention maps instead of predicted values.
        If False, returns the standard model predictions.
        Default False.
    random_state: int | None
        Setting a seed for reproducibility.
        Default None.
    **kwargs : Any
        Additional keyword arguments for the BaseEstimator class, such as
        `max_epochs`, `max_steps`, `num_sanity_val_steps`,
        `check_val_every_n_epoch`, `callbacks`, etc.

    Attributes
    ----------
    model : torch.nn.Module
        SiT network mapping input data to predictions.
    optimizer : Optimizer
        Optimizer used for training.
    lr_scheduler : LRScheduler | None
        Learning rate scheduler used for training.
    loss : MSELoss
        The MSELoss loss function used for training.

    References
    ----------
    .. [1] Dahan, Simon et al., Surface Vision Transformers: Attention-Based
    Modelling applied to Cortical Analysis, MIDL, 2022.
    .. [2] Dahan, Simon et al., Surface Vision Transformers: Flexible
    Attention-Based Modelling of Biomedical Surfaces, TMI, 2022.
    """
    def __init__(
            self,
            model: nn.Module | type[nn.Module],
            optimizer: Optimizer | type[Optimizer],
            lr_scheduler: LRScheduler | type[LRScheduler] | None = None,
            model_kwargs: dict[str, Any] | None = None,
            optimizer_kwargs: dict[str, Any] | None = None,
            lr_scheduler_kwargs: dict[str, Any] | None = None,
            metrics: dict[str, Metric]  | None = None,
            return_attention: bool = False,
            random_state: int | None = None,
            **kwargs: Any,
        ) -> None:
        super().__init__(
            random_state=random_state,
            ignore=["model", "callbacks", "metrics"],
            **(kwargs or {}),
        )

        self.model = build_instance(
            model,
            **(model_kwargs or {}),
        )
        self.attention_extractor = SiTAttentionMaps(
            model=self.model,
            module_patterns=("*attend", ),
        )
        self.optimizer = build_instance(
            optimizer,
            model.parameters,
            **(optimizer_kwargs or {}),
        )
        self.lr_scheduler = build_instance(
            lr_scheduler,
            self.optimizer,
            **(lr_scheduler_kwargs or {}),
        )
        self.loss = nn.MSELoss()
        
        self.metrics = MetricCollection(metrics or {})
        self.return_attention = return_attention

    def configure_optimizers(
            self,
        ) -> tuple[list[Optimizer], list[LRScheduler]]:
        """
        Initialize the optimizer and learning rate scheduler.

        Returns
        -------
        tuple[list[Optimizer], list[LRScheduler]
            A tuple containing two distinct lists:
            
            * optimizers : list[Optimizer]
                The first list contains the initialized optimizer for the
                network.
            * schedulers : list[LRScheduler]
                The second list contains the learning rate schedulers. Left
                empty here as no learning rate decays are applied.
        """
        if self.lr_scheduler is not None:
            return [self.optimizer], [self.lr_scheduler]
        else:
            return [self.optimizer]

    def _shared_step(
            self,
            batch: tuple[torch.Tensor, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
        """
        Shared code for training and validation steps.
        """
        # Uncpack and rearrange
        x, y = batch
        y = y.unsqueeze(1).float()
        
        # Forward pass
        preds = self.model(x)
        
        # Compute loss
        loss = self.loss(preds, y)

        return {
            "loss": loss,
            "x": x.detach(),
            "y": y.detach(),
            "preds": preds.detach(),
        }

    def training_step(
            self,
            batch: tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
        ) -> dict[str, torch.Tensor]:
        """
        Perform one training step and computes training loss and metrics.

        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor]
            Input batch data of format ``(X, y)``, where ``x``is the texture
            data of shape ``(batch, channels, vertices)`` (it contains either
            left- or right-hemisphere data, but must first be symmetrized (as
            in an xhemi-style dataset)) and ``y`` is the variable to predict
            (e.g., age) of shape ``(batch, )``.
        batch_idx : int
            Position tracking pointer of the current iteration batch.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
            - "loss": the MSE loss computed on this batch;
            - "x": the texture data of shape
              ``(batch, channels, vertices)``;
            - "y": the variable to predict of shape ``(batch, )``;
            - "preds": the predicted values of shape ``(batch, )``.
        """
        outputs = self._shared_step(batch)
        output_metrics = self.metrics(
            outputs["preds"],
            outputs["y"],
        )
        logged_metrics = {
            f"{name}/train": val
            for name, val in output_metrics.items()
        }
        logged_metrics["loss/train"] = outputs["loss"]
        self.log_dict(
            logged_metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return outputs

    def validation_step(
            self,
            batch: tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
        ) -> dict[str, torch.Tensor]:
        """
        Perform one validation step and computes validation loss and metrics.

        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor]
            Input batch data of format ```(X, y)``, where ``x``is the texture
            data of shape ``(batch, channels, vertices)`` (it contains either
            left- or right-hemisphere data, but must first be symmetrized (as
            in an xhemi-style dataset)) and ``y`` is the variable to predict
            (e.g., age) of shape ``(batch, )``.
        batch_idx : int
            Position tracking pointer of the current iteration batch.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
            - "loss": the MSE loss computed on this batch;
            - "x": the texture data of shape
              ``(batch, channels, vertices)``;
            - "y": the variable to predict of shape ``(batch, )``;
            - "preds": the predicted values of shape ``(batch, )``.
        """
        outputs = self._shared_step(batch)
        output_metrics = self.metrics(
            outputs["preds"],
            outputs["y"],
        )
        logged_metrics = {
            f"{name}/val": val
            for name, val in output_metrics.items()
        }
        logged_metrics["loss/val"] = outputs["loss"]
        self.log_dict(
            logged_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return outputs

    def predict_step(
            self,
            batch: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int = 0,
        ) -> torch.Tensor:
        """
        Infer from the trained prediction model.

        Parameters
        ----------
        batch : torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Input texture data of shape ``(batch, channels, vertices)``. It
            contains either left- or right-hemisphere data, but must first be
            symmetrized (as in an xhemi-style dataset). If batch is a tuple,
            performe inference from the first element.
        batch_idx: int
            The index of the current batch (ignored).
        dataloader_idx: int
            The index of the dataloader (ignored).
            Default 0.

        Returns
        -------
        torch.Tensor
           The predicted values of shape ``(batch, )``.
        """
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        _model = (
            self.attention_extractor
            if self.return_attention
            else self.model
        )
        return torch.tensor(_model(x))
