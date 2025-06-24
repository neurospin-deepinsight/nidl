import torch
from torch import nn
from typing import Tuple, Optional, Union, List
from einops import rearrange, repeat
from functools import partial

# Local import
from nidl.volume.utils.input_adapters import PatchEmbeddings
from nidl.volume.utils.jepa_predictor import apply_masks

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=False, dropout=0.):
        super().__init__()
        head_dim = dim // heads
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_ratio, qkv_bias, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, qkv_bias=qkv_bias, dropout=dropout),
                FeedForward(dim, int(mlp_ratio*dim), dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class VisionTransformer(nn.Module):
    """ 3D-ViT implementation adapted from [1] to handle 3D volumes as input.

    Code implementation in:
    * I-JEPA: https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py
    * DINO: https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
    * DINOv2: https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py
    * ViT-Pytorch: https://github.com/lucidrains/vit-pytorch

    All these implementations have variations in the architectural design, in particular:

    1) Positional embeddings: it is learned in DINO, DINOv2 (with TruncNormal init), 
        ViT-PyTorch (with Normal init) and it is fixed in I-JEPA (with 2d sin-cos init). 
    
    2) QKV bias inside attention layer is False for ViT-Pytorch but True for DINO, DINOv2, I-JEPA.

    3) Masking: I-JEPA implements masking inside the architecture, dropping masked embeddings (no [MASK] token required).
        DINOv2 implements a [MASK] token to be used in the loss during training. DINO and base ViT-Pytorch does not 
        implement such strategy.

    [1] An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, Dosovitskiy et al., ICLR 2021

    Parameters
    ----------

    image_size: int or Tuple[int, int, int]
        Input volume size. This is used to initialize the patch embedding layer,
        mapping the input 3D volume to a sequence of patch embeddings.
        If int, all 3 dimensions are assumed equal.
    
    patch_size: int or Tuple[int, int, int]
        Patch size over the volume size.
    
    in_channels: int, default=1
        Number of channels in the input volume.
    
    n_embedding: int, default=768
        Dimension of the patch embeddings and output dimension.
    
    depth: int, default=12
        Number of multi-head attention blocks.
    
    heads: int, default=12
        Number of heads in each multi-head attention layer.

    mlp_ratio: float, default=4.0
        Scaling factor by how much the hidden dimension in MLP is multiply 
        in attention blocks.
    
    qkv_bias: bool, default=True
        If True, add a learnable bias to the Q, K, and V projection layers in attention.

    pool: None, 'cls' or 'mean', default='cls'
        How to aggregate the output sequence of embeddings (either using CLS token or 
        with mean pooling). If None, no aggregation is applied.
    
    dropout: float, default=0.0
        Probability of dropping tokens in the multi-head attention layers.
    
    emb_dropout: float, default=0.0
        Probability of dropping some patch embeddings before going through Transformer.

    """

    def __init__(self, 
                 image_size: Union[int, Tuple[int, int, int]], 
                 patch_size: Union[int, Tuple[int, int, int]], 
                 in_channels: int=1, 
                 n_embedding: int=768, 
                 depth: int=12, 
                 heads: int=12, 
                 mlp_ratio: float=4.0,
                 qkv_bias: bool=True,
                 pool: Optional[str]='cls',
                 sincos_pos_embed: bool=False,
                 learnable_pos_embed: bool=True,
                 dropout = 0., 
                 emb_dropout = 0.):

        super().__init__()

        assert pool in {'cls', 'mean', None}, "pool type must be either 'cls' (cls token), "
        "'mean' (mean pooling) or None (no aggregation)"

        self.patch_embed = PatchEmbeddings(in_channels, patch_size=patch_size, embed_dim=n_embedding,
                                           add_pos_embed=True, sincos_pos_emb=sincos_pos_embed, 
                                           learnable_pos_emb=learnable_pos_embed, image_size=image_size)
        self.cls_pos_embedding = nn.Parameter(torch.randn(1, n_embedding))
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_embedding))
        self.mask_token = nn.Parameter(torch.randn(1, n_embedding))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(n_embedding, depth, heads, mlp_ratio, qkv_bias, dropout)
        self.pool = pool

    def forward(self, 
                img: torch.Tensor, 
                masks: Optional[Union[torch.Tensor, List[torch.Tensor]]]=None):
        """
        This method processes a 3D input image volume, optionally applies patch masks,
        adds positional embeddings and a classification token (if applicable), passes
        the sequence through a Transformer encoder, and returns the final representation
        according to the specified pooling strategy.

        Parameters
        ----------
        img : torch.Tensor
            Input tensor of shape (B, C, D, H, W), where:
            - B is the batch size,
            - C is the number of channels,
            - D, H, W are the spatial dimensions (depth, height, width).

        masks : Union[torch.Tensor, List[torch.Tensor]], optional
            Mask indices indicating which patches TO KEEP. Can be:
            - A tensor of shape (B, K), or
            - A list of such tensors, each representing a different view.
            If provided, the model will only process the masked patches.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, n_embedding) or (B, N, n_embedding),
            depending on the pooling method:
            - If `pool == 'cls'`, returns the [CLS] token embedding.
            - If `pool == 'mean'`, returns the mean-pooled token embeddings.
            - If `pool is None`, returns the full sequence of patch embeddings.

        Raises
        ------
        ValueError
            If the specified pooling strategy is not one of `'cls'`, `'mean'`, or `None`.

        Notes
        -----
        This forward pass includes optional masked patch selection, positional embedding,
        and the addition of a learnable [CLS] token. It is commonly used in masked 
        pretraining or classification tasks involving volumetric (3D) data.
        """

        x = self.patch_embed(img)

         # -- mask x
        if masks is not None:
            if isinstance(masks, torch.Tensor):
                masks = [masks]
            x = apply_masks(x, masks)

        b, _, _ = x.shape
        if self.pool == 'cls': # Add the [CLS] token
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
            cls_tokens += self.cls_pos_embedding
            x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)

        x = self.transformer(x)

        if self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'cls':
            x = x[: , 0]
        elif self.pool is None:
            pass
        else:
            raise ValueError(f"Unknown 'pool': {self.pool}")
        return x


def vit_tiny(image_size=128, patch_size=16, **kwargs):
    model = VisionTransformer(
        image_size=image_size, patch_size=patch_size, n_embedding=192, 
        depth=12, heads=3, mlp_ratio=4, qkv_bias=True, **kwargs)
    return model


def vit_small(image_size=128, patch_size=16, **kwargs):
    model = VisionTransformer(
        image_size=image_size, patch_size=patch_size, n_embedding=384, 
        depth=12, heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    return model


def vit_base(image_size=128, patch_size=16, **kwargs):
    model = VisionTransformer(
        image_size=image_size, patch_size=patch_size, n_embedding=768, 
        depth=12, heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    return model


def vit_large(image_size=128, patch_size=16, **kwargs):
    model = VisionTransformer(
        image_size=image_size, patch_size=patch_size, n_embedding=1024, 
        depth=24, heads=16, mlp_ratio=4, qkv_bias=True, **kwargs)
    return model
