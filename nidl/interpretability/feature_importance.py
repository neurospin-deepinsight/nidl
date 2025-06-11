import numpy as np
import scipy as sp
import torch
import nibabel
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import DataLoader

class SensitivityAnalysis:
    """
        Perform sensitivity analysis (via autograd; Simonyan et al. 2014) to determine the relevance of
        each input in model's output. Returns a relevance heatmap over the input data.
        It can handle classification and regression tasks:
            - For classification, the gradient is computed with respect to the  most likely class from 
              the model's output. We assume 'softmax' has already been applied.
            - For regression, the gradient is computed with respect to the output (assuming a single scalar) 
        The outputs are the concatenation of heatmaps, having the saming dimensions as input.

        Warning: in case of large dataset, this might not fit in memory.
    """
    def __init__(self, model: torch.nn.Module,
                 postprocess: str="abs",
                 task: str="regression",
                 normalize=True,
                 cuda_if_available: bool=True):
        """
        Parameters
        ----------
        model: torch.nn.Module
            Pytorch model to interpret. It is set to eval mode if Pytorch.
        postprocess: None, 'abs' or 'square', default 'abs'
            The method to postprocess the heatmap with. 'abs' is used in Simonyan et al. 2014,
            'square' is used in Montavon et al. 2018.
        task: "classification" or "regression", default="regression"
            Which task is performed. This affects how gradients are computed.
        normalize: bool, default=True
            If True, normalize spatially each heatmap to have values in [0, 1], summing to 1.
        cuda_if_available: bool, default=True
             Whether to run the computation on a cuda device (if possible)
        """

        if postprocess not in [None, 'abs', 'square']:
            raise ValueError(f"postprocess must be None, 'abs' or 'square' (got {postprocess})")
        if task not in ["classification", "regression"]:
            raise ValueError(f"'task' must be 'classification' or 'regression' (got {task})")
        
        self.model = model
        self.task = task
        self.postprocess = postprocess
        self.cuda_if_available = cuda_if_available
        self.normalize = normalize
        self.device = "cpu"

        if cuda_if_available and torch.cuda.is_available():
            self.model = model.to("cuda")
            self.device = "cuda"

    def __call__(self, dataloader: DataLoader):
        """
        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            DataLoader providing the input data as batch of (X, y).

        Returns
        ----------
        all_heatmaps: np.ndarray, shape (n_samples, *)
            Heatmaps concatenated for each sample. 
            !! Warning: this could very large for some datasets. 
        """
        self.model.eval()
        all_heatmaps = []
        for batch in dataloader:
            if not isinstance(batch[0], torch.Tensor):
                raise ValueError("Unexpected type for batch: %s"%type(batch[0]))
            # Gradients computation
            X = batch[0].to(self.device).requires_grad_(True)
            # Forward
            output = self.model(X)
            # Backward
            self.model.zero_grad()
            if self.task == "classification":
                output = output.max(dim=-1)[0]   # Take the max class probability
            else:
                output = output.squeeze() # We assume only one scalar output
            grad_output = torch.ones_like(output)
            output.backward(gradient=grad_output)
            # Heatmap
            heatmap = X.grad.detach() # same shape as input
            if self.postprocess == 'abs':
                heatmap = heatmap.abs()
            elif self.postprocess == 'square':
                heatmap = heatmap.pow(2)
            if self.normalize:
                min_max = heatmap.max() - heatmap.min()
                if min_max.abs() < 1e-8:
                    heatmap = torch.full_like(heatmap, 1 / heatmap.numel())
                else:
                    heatmap = (heatmap - heatmap.min())/min_max
                    heatmap /= heatmap.sum()
            all_heatmaps.extend(heatmap.cpu().numpy())
        all_heatmaps = np.array(all_heatmaps, dtype=np.float32)
        return all_heatmaps

class HaufeTransformation:
    """
        Computes Haufe's transformation yielding a single importance map for a given model on a dataset.
        Warning: computation of the covariance matrix is computationally expensive and it is not optimized here.

    """
    def __init__(self, model: torch.nn.Module,
                 cuda_if_available: bool=True):
        """
        Parameters
        ----------
        model: torch.nn.Module
            Pytorch model to interpret. It is set to eval mode if Pytorch.
        cuda_if_available: bool, default=True
             Whether to run the computation on a cuda device (if possible)
        """
        self.model = model
        self.cuda_if_available = cuda_if_available
        self.device = "cpu"
        if cuda_if_available and torch.cuda.is_available():
            self.model = model.to("cuda")
            self.device = "cuda"

    def __call__(self, dataloader: DataLoader):
        """
        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            DataLoader providing the input data as batch of (X, y).
        
        Returns
        ----------
        cov_: np.ndarray, shape (*num_features, *num_outputs)
            Covariance matrix between the input features and model outputs.
        """
        self.model.eval()
        features = []
        outputs = []
        dim_input = None
        for batch in dataloader:
            if not isinstance(batch[0], torch.Tensor):
                raise ValueError("Unexpected type for batch: %s"%type(batch[0]))
            dim_input = batch[0].shape
            output = self.model(batch[0].to(self.device))
            outputs.extend(output.detach().cpu().numpy().reshape(len(batch[0]), -1))
            features.extend(batch[0].detach().cpu().numpy().reshape(len(batch[0]), -1))
        cov_ = self.cov(features, outputs)
        cov_ = cov_.reshape(*dim_input[1:], -1).squeeze(-1)
        return cov_

    @staticmethod
    def cov(m1, m2):
        """
        Computes covariance matrix between m1 and m2

        Parameters
        ----------
        m1: np.ndarray, shape (n, d1)
        m2: np.ndarray, shape (n, d2)

        Returns
        ----------
        cov_: np.ndarray, shape (d1, d2)
        """
        assert len(m1) == len(m2), "Wrong matrices shape: %i != %i"%(len(m1), len(m2))
        cov = (np.array(m1) - np.mean(m1, axis=0)).T @ (np.array(m2) - np.mean(m2, axis=0)) / (len(m1) - 1)
        return cov


def plot_slices(struct_arr, num_slices=7, cmap='gray', vmin=None, vmax=None, overlay=None,
                overlay_cmap=None, overlay_vmin=None, overlay_vmax=None):
    """
    Plot equally spaced slices of a 3D image (and an overlay) along every axis
    Args:
        struct_arr (3D array or tensor): The 3D array to plot (usually from a nifti file).
        num_slices (int): The number of slices to plot for each dimension.
        cmap: The colormap for the image (default: `'gray'`).
        vmin (float): Same as in matplotlib.imshow. If `None`, take the global minimum of `struct_arr`.
        vmax (float): Same as in matplotlib.imshow. If `None`, take the global maximum of `struct_arr`.
        overlay (3D array or tensor): The 3D array to plot as an overlay on top of the image. Same size as `struct_arr`.
        overlay_cmap: The colomap for the overlay (default: `alpha_to_red_cmap`).
        overlay_vmin (float): Same as in matplotlib.imshow. If `None`, take the global minimum of `overlay`.
        overlay_vmax (float): Same as in matplotlib.imshow. If `None`, take the global maximum of `overlay`.
    """
    if overlay_cmap is None:
        alpha_to_red_cmap = np.zeros((256, 4))
        alpha_to_red_cmap[:, 0] = 0.8
        alpha_to_red_cmap[:, -1] = np.linspace(0, 1, 256)  # cmap.N-20)  # alpha values
        alpha_to_red_cmap = mpl.colors.ListedColormap(alpha_to_red_cmap)
        overlay_cmap = alpha_to_red_cmap

    if vmin is None:
        vmin = struct_arr.min()
    if vmax is None:
        vmax = struct_arr.max()
    if overlay_vmin is None and overlay is not None:
        overlay_vmin = overlay.min()
    if overlay_vmax is None and overlay is not None:
        overlay_vmax = overlay.max()
    print(vmin, vmax, overlay_vmin, overlay_vmax)

    fig, axes = plt.subplots(3, num_slices, figsize=(15, 6))
    intervals = np.asarray(struct_arr.shape) / num_slices

    for axis, axis_label in zip([0, 1, 2], ['x', 'y', 'z']):
        for i, ax in enumerate(axes[axis]):
            i_slice = int(np.round(intervals[axis] / 2 + i * intervals[axis]))
            # print(axis_label, 'plotting slice', i_slice)

            plt.sca(ax)
            plt.axis('off')
            plt.imshow(sp.ndimage.rotate(np.take(struct_arr, i_slice, axis=axis), 90), vmin=vmin, vmax=vmax,
                       cmap=cmap, interpolation=None)
            plt.text(0.03, 0.97, '{}={}'.format(axis_label, i_slice), color='white',
                     horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

            if overlay is not None:
                plt.imshow(sp.ndimage.rotate(np.take(overlay, i_slice, axis=axis), 90), cmap=overlay_cmap,
                           vmin=overlay_vmin, vmax=overlay_vmax, interpolation=None)


def get_relevance_per_area(area_masks, relevance_map, normalize=True, merge_hemisphere=False):
    relevances = dict()
    for area, area_mask in area_masks.items():
        relevances[area] = np.sum(relevance_map * area_mask)
    if normalize:
        normalizing_cst = np.array(list(relevances.values())).sum()
        if normalizing_cst > 0:
            for area in area_masks:
                relevances[area] /= normalizing_cst  # make all areas sum to 1
        else:
            print("Warning: relevance scores sum to 0", flush=True)
            for area in area_masks:
                relevances[area] = 1./len(area_masks)
    if merge_hemisphere:
        # Merge left and right areas.
        for area in area_masks:
            if re.match(r"\w*_L$", area):
                area_RL = re.match(r"(\w*)_L$", area)[1] # extract area name without "_L"
                relevances[area_RL] = relevances[area_RL+"_L"] + relevances[area_RL+"_R"]
                del(relevances[area_RL+"_L"], relevances[area_RL+"_R"])
    return sorted(relevances.items(), key=lambda b:b[1], reverse=True)


def parse_atlas_mapping(path_to_mapping):
    # From a .txt file, parse each line and get a dict(index: str) indicating the region
    with open(path_to_mapping, "r") as f:
        all_lines = f.readlines()
        atlas_map = {int(l.split(" ")[0]): l.split(" ")[1] for l in all_lines}
    return atlas_map

def resize_image(img, size, interpolation=0):
    """Resize img to size. Interpolation between 0 (no interpolation) and 5 (maximum interpolation)."""
    zoom_factors = np.asarray(size) / np.asarray(img.shape)
    return sp.ndimage.zoom(img, zoom_factors, order=interpolation)

def get_brain_area_masks(data_size, path_to_atlas, path_to_mapping_atlas, transforms=None):
    brain_map = nibabel.load(path_to_atlas).get_fdata()
    brain_areas = np.unique(brain_map)[1:]  # omit background
    mapping_atlas = parse_atlas_mapping(path_to_mapping_atlas)
    brain_areas_masked = dict()
    for area in brain_areas:
        area_mask = np.zeros_like(brain_map)
        area_mask[brain_map == area] = 1
        area_mask = resize_image(area_mask, data_size, interpolation=0).astype(np.bool)
        if transforms is not None:
            area_mask = transforms(area_mask).astype(np.bool)
        brain_areas_masked[mapping_atlas[area]] = area_mask
    return brain_areas_masked
