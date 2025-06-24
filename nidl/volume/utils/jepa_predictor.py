import torch.nn as nn
import torch
from torch.nn.init import trunc_normal_
import math
from einops import rearrange
from typing import Union, Tuple, List
# Local library
from .input_adapters import build_3d_sincos_posemb, triplet

""" Define the predictor module (lightweight ViT-3D) in the JEPA model. """


def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x


def apply_masks(x: torch.Tensor, masks: List[torch.Tensor]):
    """
    Applies patch-wise masks to a batched input tensor and concatenates the results across views.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, N, D), where:
        - B is the batch size,
        - N is the number of patches (tokens),
        - D is the feature dimension.
    
    masks : List[torch.Tensor]
        List of M tensors, each of shape (B, K), where:
        - K is the number of patches to keep,
        - Each tensor contains the indices of the patches to retain in each sample of the batch.

    Returns
    -------
    torch.Tensor
        Tensor of shape (B * M, K, D), representing the masked outputs for each masking view.
        The batch is repeated M times, once for each set of mask indices.

    Notes
    -----
    This function is typically used in masked image modeling tasks like JEPA or MAE,
    where multiple masking views are applied to the same input batch to produce
    different sub-sampled contexts for prediction.
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformerPredictor(nn.Module):
    """ Lightweight Vision Transformer taking as input a sequence of tokens. """
    def __init__(
        self,
        patch_size: Union[int, Tuple[int,int,int]] = 16,
        image_size: Union[int, Tuple[int,int,int]] = 128,
        embed_dim=192,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02
    ):
        super().__init__()
        self.patch_size = triplet(patch_size)
        self.image_size = triplet(image_size)
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # --
        P_H = max(1, self.patch_size[0])
        P_W = max(1, self.patch_size[1])
        P_D = max(1, self.patch_size[2])
        h = self.image_size[0] // P_H
        w = self.image_size[1] // P_W
        d = self.image_size[2] // P_D
        num_patches = h * w * d
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim),
                                                requires_grad=False)
        predictor_pos_embed = rearrange(
            build_3d_sincos_posemb(h=h, w=w, d=d, embed_dim=predictor_embed_dim),
            'b d nh nw nd -> b (nh nw nd) d'
            )
        self.predictor_pos_embed.data.copy_(predictor_pos_embed)
        # --
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks_x, masks):
        """
        Perform forward pass of the JEPA predictor to infer masked token representations 
        from contextual input tokens.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B * V, N, D), where B is the batch size, V is 
            the number of context masks (=len(masks_x)), N is the number of visible 
            patches (tokens), and D is the feature dimension from the context encoder.
        
        masks_x : Union[torch.Tensor, List[torch.Tensor]]
            List of tensors (or single tensor) containing indices of context tokens retained 
            from the input `x` by the context encoder.
            NB: usually, only one context mask is given during JEPA training, 
                i.e. len(masks_x)=1

        masks : Union[torch.Tensor, List[torch.Tensor]]
            List of tensors (or single tensor) containing indices of tokens to predict.
            The length M is the number of targets to predict for each context. 

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B * V * M, N_masked, D'), where M is the number of 
            target masks, V the number of context masks, N_masked is the number of target 
            tokens per sample, and D' is the predictor output feature dimension.
            This tensor contains predicted features for each masked token.

        Raises
        ------
        AssertionError
            If `masks` or `masks_x` is None.
        
        """
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        # -- Batch Size
        B = len(x) // len(masks_x)

        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)

        # -- add positional embedding to x tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_x)

        _, N, D = x.shape

        # -- concat mask tokens to x
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1) # shape [B, N, D]
        pos_embs = apply_masks(pos_embs, masks) # shape [B, K, D]
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x)) # shape [B * len(masks_x), K, D]
        # -- create mask tokens (learnable)
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # -- add positional embeddings to mask tokens
        pred_tokens += pos_embs
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = x[:, N:]
        x = self.predictor_proj(x)

        return x