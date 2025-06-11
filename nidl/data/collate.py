import torch
import torch.nn as nn
import numpy as np
from typing import Sequence, Tuple, Union, Any, Optional, List
from sklearn.utils.validation import check_array
from multiprocessing import Value
import math
# Local imports
from nidl.transform import Transform, Identity

class BaseCollateFunction(nn.Module):
    """Base class for all collate functions. It handles the convertion from a batch of 
    input data (with eventually heterogeneous types) to homogeneous torch.Tensor. 

    The basic collate_fn takes a batch of data as input, concatenate them in the 1st dimension
    and outputs a torch.Tensor with same length as input.

    Input batch is expected to be a sequence of tuples (X, y) and output is a pair of torch.Tensor.
    
    Examples
    ----------
    >>> # define the collate function
    >>> collate_fn = BaseCollateFunction()
    >>> # input is a sequence of tuples (here, batch_size = 1)
    >>> input = [(img, 0)]
    >>> output = collate_fn(input)
    >>> # output consists of a batch of b samples (b==1 here), containing the image and the label
    >>> img, label = output[0], output[1]
    """

    def __init__(self, transform: Transform=Identity(), 
                 target_transform: Optional[Transform]=None):
        """
        Parameters
        ----------
        transform: Transform, default=Identity()
            Transformation to apply to the input data.

        target_transform: Transform or None, default=None
            Transformation to apply to the target labels. 
            If None, not transformation is applied.
        """
        super(BaseCollateFunction, self).__init__()
        self.transform = transform
        self.target_transform = target_transform

    def forward(self, batch: Sequence[Tuple[np.ndarray, Any]]):
        """Turns a batch of tuples into a pair of torch.Tensor.

        Parameters
        ----------
        batch:
            A sequence of tuples of (data, label) where data is a np.ndarray
            and label is 'numeric'. label can be eventually None.
            All data must be concatenable, that is, they all should
            have same shape.
        
        Returns
        ----------                
        (data, labels): torch.Tensor, torch.Tensor or None
        """
        batch_size = len(batch)

        self._check_batch(batch)

        X = check_array([self.transform(batch[i][0]) for i in range(batch_size)], allow_nd=True)
        y = [self.target_transform(batch[i][1]) if self.target_transform is not None else batch[i][1] 
             for i in range(batch_size)]
        if batch_size > 0:
            if y[0] is not None:
                y = check_array(y, allow_nd=True, ensure_2d=False)
            elif not np.all([yi is None for yi in y]):
                raise ValueError("Inconsistent labels: only some of them are None.")
            else:
                y = None

        # convert input data to torch.Tensor
        X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            y = torch.tensor(y, dtype=torch.float32)
        return X, y

    def _check_batch(self, batch: Sequence[Tuple[np.ndarray, Any]]):
        """
        Ensures batch is a sequence of tuples with two elements in each tuple.
        It does not check elements type.
        """
        if not isinstance(batch, Sequence):
            raise ValueError("batch must be a sequence of tuples (data, label).")

        for elem in batch:
            if not isinstance(elem, Tuple) or len(elem) != 2:
                raise ValueError("batch must be a sequence of tuples (data, label).")


class TwoViewsCollateFunction(BaseCollateFunction):
    """Collate function generating two views of each sample by applying the same data augmentation.
    3 tensors are returned:
    - V1: first view
    - V2: second view
    - y: labels (if any, can be None)

    Examples
    ----------
    >>> from neuroclav.transforms import RandomResizedCrop
    >>> # define the collate function
    >>> collate_fn = TwoViewsCollateFunction(transform=RandomResizedCrop(128, scale=(0.5, 1)))
    >>> # input is a sequence of tuples (here, batch_size = 1)
    >>> input = [(img, 0)]
    >>> output = collate_fn(input)
    >>> # output consists of a batch of b samples (b==1 here),
    >>> # containing two views of each sample and the labels
    >>> v1, v2, label = output[0], output[1], output[2]
    """
    def forward(self, batch: Sequence[Tuple[np.ndarray, Any]]):
        """Turns a batch of tuples into a triplet of torch.Tensor.

        Parameters
        ----------
        batch:
            A sequence of tuples of (data, label) where data is an array or Tensor
            and label is 'numeric'. label can be eventually None.
            All data must be concatenable, that is, they all should have same shape.
        
        Returns
        ----------
        (V1, V2, y): torch.Tensor, torch.Tensor, torch.Tensor or None
        """
        batch_size = len(batch)

        self._check_batch(batch)
        V1 = check_array([self.transform(batch[i][0]) for i in range(batch_size)], allow_nd=True)
        V2 = check_array([self.transform(batch[i][0]) for i in range(batch_size)], allow_nd=True)

        y = [self.target_transform(batch[i][1]) if self.target_transform is not None else batch[i][1] 
             for i in range(batch_size)]
        
        if batch_size > 0:
            if y[0] is not None:
                y = check_array(y, allow_nd=True, ensure_2d=False)
            elif not np.all([yi is None for yi in y]):
                raise ValueError("Inconsistent labels: only some of them are None.")
            else:
                y=None

        # convert input data to torch.Tensor
        V1 = torch.tensor(V1, dtype=torch.float32)
        V2 = torch.tensor(V2, dtype=torch.float32)
        if y is not None:
            y = torch.tensor(y, dtype=torch.float32)
        return V1, V2, y


class MaskCollateFunction(BaseCollateFunction):
    """Collate function that generates:
        1) the batch of images with eventual labels (can be None)
        2) 'num_enc_masks' masks per sample given to a context encoder with random scale and unit aspect ratio
        3) 'num_pred_masks' masks per sample given to a target predictor with random scale and random aspect ratio
        
        It handles 3D volumes as input and generates flattened 3D masks indices as 1D vectors.
        This collate function is adapted from the original JEPA implementation in :
        https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py
        
        By default, The masks generated for the context encoder should not overlap with the masks generated for the target predictor.

        Note: input 3D volume is seen as an ordered sequence of small 3D patches and the masks are defined over this grid of patches
        (following ViT view of an image). As such, mask indices are defined over this grid composed of 3D patches and NOT over input voxels.

        Examples
        ----------
        >>> # Define the collate function
        >>> collate_fn = MaskCollateFunction()
        >>> # Input is a sequence of tuples (here, batch_size = 1)
        >>> input = [(img, 0)]
        >>> output = collate_fn(input)
        >>> # Output consists of a MaskBatch of b samples (b==1 here),
        >>> X, y = output[0], output[1]
        >>> # List of 'num_enc_masks' and 'num_pred_masks' are available too
        >>> masks_enc, masks_pred = output[2], output[3]
    """
    def __init__(
        self,
        input_size=(128, 128, 128),
        patch_size=16,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        num_enc_masks=1,
        num_pred_masks=4,
        min_keep=4,
        allow_overlap=False,
        transform: Transform=Identity(),
        target_transform: Optional[Transform]=None):
        """
        Parameters
        ----------
        input_size: (int, int, int) or int, default=(128, 128, 128)
            Size of the input volumes (height, width, depth). 
            Must be 1D or 3D. If 1D, the same size is used for all dimensions.

        patch_size: (int, int, int) or int, default=16
            Size of the input patches given to the context encoder and predictor.

        enc_mask_scale: (float, float), default=(0.85, 1.0)
            Range of the scale of the masks given to the context encoder.

        pred_mask_scale: (float, float), default=(0.15, 0.2)
            Range of the scale of the masks given to the target predictor.

        aspect_ratio: (float, float), default=(0.75, 1.5)
            Range of the aspect ratio of the masks given to the target predictor.
        
        num_enc_masks: int, default=1
            Number of masks generated for the context encoder.

        num_pred_masks: int, default=4
            Number of masks generated for the target predictor.

        min_keep: int, default=4
            Minimum number of patches to keep in the context encoder mask.

        allow_overlap: bool, default=False
            Whether to allow overlap between the masks generated for the context encoder and the target predictor.
        
        transform: Transform, default=Identity()
            Transformation to apply to the input data.
        
        target_transform: Optional[Transform], default=None
            Transformation to apply to the target labels.
        """

        super().__init__(transform=transform, target_transform=target_transform)
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 3
        if not isinstance(patch_size, tuple):
            patch_size = (patch_size, ) * 3
        if len(input_size) != 3 or len(patch_size) != 3:
            raise ValueError("input_size and patch_size must be 3D tuples.")
        self.input_size = input_size
        self.patch_size = patch_size
        self.height, self.width, self.depth = input_size[0] // patch_size[0], \
            input_size[1] // patch_size[1], input_size[2] // patch_size[2]
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.num_enc_masks = num_enc_masks
        self.num_pred_masks = num_pred_masks
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        # Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * self.depth * mask_scale)
        # Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # Compute block height, width and depth (given scale and aspect-ratio)
        axis1 = int(round(math.pow(max_keep * aspect_ratio, 1/3)))
        axis2 = int(round(math.pow(max_keep / aspect_ratio, 1/3)))
        axis3 = int(round(math.pow(max_keep, 1/3)))
        dims = torch.tensor([axis1, axis2, axis3])
        perm = torch.randperm(3, generator=generator)
        h, w, d = dims[perm].tolist()
    
        # Constrain block size to be smaller than the input size
        h = min(h, self.height - 1)
        w = min(w, self.width - 1)
        d = min(d, self.depth - 1)

        return (h, w, d)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        # Generates a mask at a random location as a tensor of indices
        # and its complementary as a 3D tensor. 
        h, w, d = b_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left-front corner
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            front = torch.randint(0, self.depth - d, (1,))

            mask = torch.zeros((self.height, self.width, self.depth), dtype=torch.int32)
            mask[top:top+h, left:left+w, front:front+d] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    print(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width, self.depth), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w, front:front+d] = 0
        # --
        return mask, mask_complement

    def forward(self, batch: Sequence[Tuple[np.ndarray, Any]]):
        """Turns a batch of tuples into a MaskBatch object containing torch.Tensor and 2 
        list of masks (as 1D tensors): one for the context encoder and the other for the target predictor.

        Strategy for creating the masks:
            1. sample context block (size + location) using seed
            2. sample target block (size) using seed
            3. sample several context block locations for each image (w/o seed)
            4. sample several target block locations for each image (w/o seed)
            5. return context mask and target mask

        Parameters
        ----------
        batch:
            A sequence of tuples of (data, label) where data is an array or Tensor
            and label is 'numeric'. label can be eventually None.
            All data must be concatenable, that is, they all should have same shape.
            Additionally, input data size must match 'self.input_size'.
        
        Returns
        ----------
        (data, labels, masks_enc, masks_pred): torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]
        """
        B = len(batch)

        base_batch = super().forward(batch)
        
        if base_batch[0].shape[-3:] != tuple(self.input_size):
            raise ValueError(f"Unexpected input size for the batch (expected "
                             f"{self.input_size} but got {base_batch[0].shape[-3:]})")

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width * self.depth
        min_keep_enc = self.height * self.width * self.depth

        for _ in range(B):
            masks_p, masks_C = [], []
            for _ in range(self.num_pred_masks):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            
            if self.allow_overlap:
                acceptable_regions= None

            masks_e = []
            for _ in range(self.num_enc_masks):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        # List of masks with size num_pred_masks x (batch_size, min_keep_pred)
        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        
        # List of masks with size num_enc_masks x (batch_size, min_keep_enc)
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return base_batch[0], base_batch[1], collated_masks_enc, collated_masks_pred

 