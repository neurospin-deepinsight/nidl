from typing import List, Optional, Tuple
import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    """Base class for all projection and prediction heads.

    Args:
        blocks:
            List of tuples, each denoting one block of the projection head MLP.
            Each tuple reads (in_features, out_features, batch_norm_layer,
            non_linearity_layer).

    Examples:
        >>> # the following projection head has two blocks
        >>> # the first block uses batch norm an a ReLU non-linearity
        >>> # the second block is a simple linear layer
        >>> projection_head = ProjectionHead([
        >>>     (256, 256, nn.BatchNorm1d(256), nn.ReLU()),
        >>>     (256, 128, None, None)
        >>> ])

    """

    def __init__(
            self,
            blocks: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]]
    ):
        super(ProjectionHead, self).__init__()

        layers = []
        for input_dim, output_dim, batch_norm, non_linearity in blocks:
            use_bias = not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Computes one forward pass through the projection head.

        Args:
            x:
                Input of shape bsz x num_ftrs.

        """
        return self.layers(x)


class BarlowTwinsProjectionHead(ProjectionHead):
    """Projection head used for Barlow Twins.

    "The projector network has three linear layers, each with 8192 output
    units. The first two layers of the projector are followed by a batch
    normalization layer and rectified linear units." [0]

    [0]: 2021, Barlow Twins, https://arxiv.org/abs/2103.03230

    """

    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dim: int = 8192,
                 output_dim: int = 8192):
        super(BarlowTwinsProjectionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])


class VICRegProjectionHead(ProjectionHead):
    """Projection head used for VICReg.

    "The expander hφ is composed of two fully-connected layers with batch normalization (BN) 
     and ReLU, and a third linear layer. The sizes of all 3 layers were set to 8192. As with 
     Barlow Twins, performance improves when the size of the expander layers is larger than the
     dimension of the representation." [0]

    [0]: 2022, VICReg, https://arxiv.org/pdf/2105.04906

    """

    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dim: int = 8192,
                 output_dim: int = 8192):
        super(VICRegProjectionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])


class BYOLProjectionHead(ProjectionHead):
    """Projection head used for BYOL.

    "This MLP consists in a linear layer with output size 4096 followed by
    batch normalization, rectified linear units (ReLU), and a final
    linear layer with output dimension 256." [0]

    [0]: BYOL, 2020, https://arxiv.org/abs/2006.07733

    """
    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dim: int = 4096,
                 output_dim: int = 256):
        super(BYOLProjectionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])


class MoCoProjectionHead(ProjectionHead):
    """Projection head used for MoCo.

    "(...) we replace the fc head in MoCo with a 2-layer MLP head (hidden layer
    2048-d, with ReLU)" [0]

    [0]: MoCo, 2020, https://arxiv.org/abs/1911.05722

    """

    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dim: int = 2048,
                 output_dim: int = 128):
        super(MoCoProjectionHead, self).__init__([
            (input_dim, hidden_dim, None, nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])


class NNCLRProjectionHead(ProjectionHead):
    """Projection head used for NNCLR.

    "The architectureof the projection MLP is 3 fully connected layers of sizes
    [2048,2048,d] where d is the embedding size used to apply the loss. We use
    d = 256 in the experiments unless otherwise stated. All fully-connected
    layers are followed by batch-normalization [36]. All the batch-norm layers
    except the last layer are followed by ReLU activation." [0]

    [0]: NNCLR, 2021, https://arxiv.org/abs/2104.14548

    """
    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dim: int = 2048,
                 output_dim: int = 256):
        super(NNCLRProjectionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, nn.BatchNorm1d(output_dim), None),
        ])


class SimCLRProjectionHead(ProjectionHead):
    """Projection head used for SimCLR.

    "We use a MLP with one hidden layer to obtain zi = g(h) = W_2 * σ(W_1 * h)
    where σ is a ReLU non-linearity." [0]

    [0] SimCLR, 2020, https://arxiv.org/abs/2002.05709

    """
    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dim: int = 2048,
                 output_dim: int = 128):
        super(SimCLRProjectionHead, self).__init__([
            (input_dim, hidden_dim, None, nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])


class yAwareProjectionHead(ProjectionHead):
    """Projection head used for yAware.

    "[...] zθ2 is a vanilla multilayer perceptron as in [SimCLR, 2020]" [0]

    [0] yAware, 2021, https://arxiv.org/abs/2106.08808

    """
    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dim: int = 512,
                 output_dim: int = 128):
        super(yAwareProjectionHead, self).__init__([
            (input_dim, hidden_dim, None, nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])


class SimSiamProjectionHead(ProjectionHead):
    """Projection head used for SimSiam.

    "The projection MLP (in f) has BN applied to each fully-connected (fc)
    layer, including its output fc. Its output fc has no ReLU. The hidden fc is
    2048-d. This MLP has 3 layers." [0]

    [0]: SimSiam, 2020, https://arxiv.org/abs/2011.10566

    """
    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dim: int = 2048,
                 output_dim: int = 2048):
        super(SimSiamProjectionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, nn.BatchNorm1d(output_dim, affine=False), None),
        ])


class SMoGProjectionHead(ProjectionHead):
    """Projection head used for SMoG.

    "The two kinds of head are both a two-layer MLP and their hidden layer is
    followed by a BatchNorm [28] and an activation function. (...) The output
    layer of projection head also has BN" [0]

    [0]: SMoG, 2022, https://arxiv.org/pdf/2207.06167.pdf

    """
    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dim: int = 2048,
                 output_dim: int = 128):
        super(SMoGProjectionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(2048), nn.ReLU()),
            (hidden_dim, output_dim, nn.BatchNorm1d(128, affine=False), None)
        ])


class SwaVProjectionHead(ProjectionHead):
    """Projection head used for SwaV.

    [0]: SwAV, 2020, https://arxiv.org/abs/2006.09882
    """
    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dim: int = 2048,
                 output_dim: int = 128):
        super(SwaVProjectionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])


class MSNProjectionHead(ProjectionHead):
    """Projection head for MSN [0].

    "We train with a 3-layer projection head with output dimension 256 and
    batch-normalization at the input and hidden layers.." [0]
    Code inspired by [1].

    - [0]: Masked Siamese Networks, 2022, https://arxiv.org/abs/2204.07141
    - [1]: https://github.com/facebookresearch/msn

    Attributes:
        input_dim:
            Input dimension, default value 768 is for a ViT base model.
        hidden_dim:
            Hidden dimension.
        output_dim:
            Output dimension.
    """
    def __init__(
            self,
            input_dim: int = 768,
            hidden_dim: int = 2048,
            output_dim: int = 256,
    ):
        super().__init__(blocks=[
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.GELU()),
            (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.GELU()),
            (hidden_dim, output_dim, None, None),
        ])
