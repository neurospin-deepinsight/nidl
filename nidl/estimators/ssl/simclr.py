##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from ...losses import InfoNCE
from ..base import BaseEstimator, TransformerMixin


class SimCLR(TransformerMixin, BaseEstimator):
    r"""SimCLR implementation.

    At each iteration, we get for every data x two differently augmented
    versions, which we refer to as x_i and x_j. Both of these images are
    encoded into a one-dimensional feature vector, between which we want to
    maximize similarity which minimizes it to all other data in the batch.
    The encoder network is split into two parts: a base encoder network f(.),
    and a projection head g(.). The base network is usually a deep CNN or SCNN,
    and is responsible for extracting a representation vector from the
    augmented data examples. Let's denote the representations obtained from the
    encoder h=f(x). The projection head g(.) maps the representation h into a
    space where we apply the contrastive loss, i.e., compare similarities
    between vectors. In the original SimCLR paper g(.) was defined as a
    two-layer MLP with ReLU activation in the hidden layer. Note that in the
    follow-up paper, SimCLRv2, the authors mention that larger/wider MLPs can
    boost the performance considerably.

    After finishing the training with contrastive learning, we will remove
    the projection head g(.), and use f(.) as a pretrained feature extractor.
    The representations z that come out of the projection head g(.) have been
    shown to perform worse than those of the base network f(.) when
    finetuning the network for a new task. This is likely because the
    representations z are trained to become invariant to many features that
    can be important for downstream tasks. Thus, g(.) is only needed for the
    contrastive learning stage.

    Now that the architecture is described, let's take a closer look at how we
    train the model. As mentioned before, we want to maximize the similarity
    between the representations of the two augmented versions of the same
    image, i.e., z_i and z_j, while minimizing it to all other examples in the
    batch. SimCLR thereby applies the InfoNCE loss, originally proposed by
    Aaron van den Oord et al. for contrastive learning. In short, the InfoNCE
    loss compares the similarity of z_i and z_j to the similarity of z_i to
    any other representation in the batch by performing a softmax over the
    similarity values. The loss can be formally written as:

    .. math::

        \ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i,z_j)/\tau)}{
                     \sum_{k=1}^{2N}\mathbb{1}_{[k\neq i]}
                        \exp(\text{sim}(z_i,z_k)/\tau)}
                   = -\text{sim}(z_i,z_j)/\tau
                     +\log\left[\sum_{k=1}^{2N}\mathbb{1}_{[k\neq i]}
                        \exp(\text{sim}(z_i,z_k)/\tau)\right]

    The function \text{sim} is a similarity metric, and the hyperparameter
    \tau is called temperature determining how peaked the distribution is.
    Since many similarity metrics are bounded, the temperature parameter
    allows us to balance the influence of many dissimilar image patches versus
    one similar patch. The similarity metric that is used in SimCLR is cosine
    similarity, as defined below:

    .. math::

        \text{sim}(z_i,z_j) = \frac{z_i^\top \cdot z_j}{||z_i||\cdot||z_j||}

    The maximum cosine similarity possible is 1, while the minimum is -1. In
    general, we will see that the features of two different images will
    converge to a cosine similarity around zero since the minimum, -1, would
    require z_i and z_j to be in the exact opposite direction in all feature
    dimensions, which does not allow for great flexibility.

    Alternatively to performing the validation on the contrastive learning
    loss as well, we could also take a simple, small downstream task, and
    track the performance of the base network f(.) on that.

    Parameters
    ----------
    encoder: nn.Module
        the encoder f(.). It must store the size of the encoded one-dimensional
        feature vector in a `latent_size` parameter.
    hidden_dims: list of str
        the projector g(.) MLP architecture.
    lr: float
        the learning rate.
    temperature: float
        the SimCLR loss temperature parameter.
    weight_decay: float
        the Adam optimizer weight decay parameter.
    max_epochs: int, default=None
        optionaly, use a CosineAnnealingLR scheduler.
    random_state: int, default=None
        setting a seed for reproducibility.
    kwargs: dict
        Trainer parameters.

    Attributes
    ----------
    f
        a :class:`~torch.nn.Module` containing the encoder.
    g
        a :class:`~torch.nn.Module` containing the projection head.

    Notes
    -----
    A batch of data must contains two elements: two tensors with contrasted
    images, and a list of tensors containing auxiliary variables.
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_dims: Sequence[str],
        lr: float,
        temperature: float,
        weight_decay: float,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            random_state=random_state, ignore=["encoder"], **kwargs
        )
        assert self.hparams.temperature > 0.0, (
            "The temperature must be a positive float!"
        )
        assert hasattr(encoder, "latent_size"), (
            "The encoder must store the size of the encoded one-dimensional "
            "feature vector in a `latent_size` parameter!"
        )
        self.f = encoder
        self.g = torchvision.ops.MLP(
            in_channels=self.f.latent_size,
            hidden_channels=hidden_dims,
            activation_layer=nn.ReLU,
            inplace=True,
            bias=True,
            dropout=0.0,
        )
        self.g = nn.Sequential(
            *[
                layer
                for layer in self.g.children()
                if not isinstance(layer, nn.Dropout)
            ]
        )

    def configure_optimizers(self):
        """Declare a :class:`~torch.optim.AdamW` optimizer and, optionnaly
        (``max_epochs`` is defined), a
        :class:`~torch.optim.lr_scheduler.CosineAnnealingLR` learning-rate
        scheduler.
        """
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if (
            hasattr(self.hparams, "max_epochs")
            and self.hparams.max_epochs is not None
        ):
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs,
                eta_min=(self.hparams.lr / 50),
            )
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]

    def info_nce_loss(
        self, batch: tuple[torch.Tensor, torch.Tensor], mode: str
    ):
        """Compute and log the InfoNCE loss using
        :class:`~nidl.losses.InfoNCE`.
        """
        # Encode all images
        n_samples = len(batch[0])
        imgs = torch.cat(batch, dim=0)
        feats = self.g(self.f(imgs))
        # Calculate loss
        nll = InfoNCE(self.hparams.temperature)(
            feats[:n_samples], feats[n_samples:]
        )
        # Logging loss
        self.log(mode + "_loss", nll, prog_bar=True)
        return nll

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        self.info_nce_loss(batch, mode="val")

    def transform_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        return self.f(batch)
