from torch.utils.data.dataset import Dataset
import os, re
import nibabel
import pandas as pd
import numpy as np
import json
import subprocess
import shutil
from typing import Callable, Tuple, Union

class OpenBHB(Dataset):
    """
        OpenBHB dataset [1]. It includes brain imaging data from 10 datasets. 

        If not already present, data are downloaded automatically in the root directory (~320GB).

        Briefly, OpenBHB contains 3227 T1 MRI in train and 757 T1 MRI in validation. Each volume has been 
        pre-processed with 3 different pipelines (voxel-based morphometry with CAT12 SPM, surface-based 
        morphometry with FreeSurfer, and "quasi-raw') to extract 6 pre-processed Numpy arrays. Raw T1 MRI 
        (without any pre-processing) is also available for all volumes.

        Target tasks include age regression, sex classification and site classification. 

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.read_csv("participants.tsv", sep="\t")
        >>> df['age'].describe()
        count    3984.000000
        mean       24.922390
        std        14.287559
        min         5.900000
        25%        19.000000
        50%        21.000000
        75%        26.000000
        max        88.000000
        >>> df = pd.read_csv("qc.tsv", sep="\t")
        >>> df.describe()
                reconall-euler  cat12vbm-ncr  cat12vbm-iqr  quasiraw-corr
        count     3984.000000   3984.000000   3984.000000    3984.000000
        mean       -32.446034      2.165936      2.197363       0.786914
        std         26.068232      0.456252      0.306228       0.135918
        min       -216.000000      0.773947      1.325781       0.502124
        25%        -39.000000      1.859899      1.954573       0.665743
        50%        -24.000000      2.233487      2.184664       0.757415
        75%        -16.000000      2.488782      2.437508       0.929191
        max          1.000000      3.895208      3.575637       1.000000

        [1] OpenBHB: a Large-Scale Multi-Site Brain MRI Data-set for Age Prediction and Debiasing, Dufumier et al., NeuroImage 2022
    """

    REPO_URL = "https://huggingface.co/datasets/benoit-dufumier/openBHB"

    def __init__(self,
                 root: str, 
                 modality: Union[str, Tuple[str]]='vbm', 
                 target: str='age',
                 split: str='train', 
                 transforms: Callable=None,
                 target_transforms: Callable=None,
                 verbose: bool=False):
        """
        Parameters
        ----------
        root: str
            Path to the root data directory where the HF repository is stored.

        modality: str or Tuple[str] in {"vbm", "quasiraw", "vbm_roi", 
            "fs_desikan_roi", "fs_destrieux_roi", "fs_xhemi"}

            Which modality to load for each brain image. When 'modality' is a tuple (multimodal OpenBHB), 
            a dict is generated in __getitem__ where keys are the modality names 
            and values are the numpy arrays. 
            
            Available modalities:
                "vbm" = Voxel Brain Morphometry, it is a whole-brain 3D T1w image. 
                "quasiraw" = Minimal preprocessing, it is a whole-brain 3D T1 image.
                "vbm_roi" = Gray matter volume per region computed on the Neuromorphometric atlas (284 regions).
                "fs_desikan_roi" = FreeSurfer measures (surface area, GM volume, thickness, curvature) 
                    computed over Desikan atlas (68 regions)
                "fs_destrieux_roi" = FreeSurfer measures computed over Destrieux atlas (148 regions)
                "fs_xhemi"= FreeSurfer measures computed on the surface mesh fsaverage7. 

        target: str  in {'age', 'sex' or 'site'}
            Target value to return with each image.

        split: str in {'train', 'val', 'internal_val', 'external_val'}
            The dataset split. 'val' is the union of 'internal val' and 'external val'.
            'internal val' = images acquired on same MRI than training (in-domain)
            'external val' = images acquired on different MRI than training (out-of-domain)

        transforms: callable or None, default=None
            A function/transform that takes in a brain image and returns a transformed version.
            Exact input depends on "modality" (can be 3D image, 1D vector or a dictionary)

        target_transforms: callable or None, default=None
            A function/transform that takes in the target and returns a transformed version.
        
        verbose: bool, default=False
            If True, print checks and downloading steps if required.
        """

        root = os.path.expanduser(root)
        valid_modalities = ['vbm', 'vbm_roi', 'quasiraw', 'fs_desikan_roi', 'fs_destrieux_roi', 'fs_xhemi']
        valid_splits = ["train", "val", "internal_val", "external_val"]
        valid_targets = ["age", "sex", "site"]

        if isinstance(modality, str):
            modality = (modality,)
        
        if split not in valid_splits:
            raise ValueError(f"'split' must be in {valid_splits}")

        for mod in modality:
            if mod not in valid_modalities:
                raise ValueError(f"'modality' must be in {valid_modalities}")
        
        if target not in valid_targets:
            raise ValueError(f"'target' must be in {valid_targets}")

        self.root = root
        self.modality = modality
        self.split = split
        self.target = target
        self.transforms = transforms
        self.target_transforms = target_transforms

        self.fetch_dataset(root, verbose=verbose)
        self.samples = self.make_dataset()
        self._shape = self._get_shape()     

    def make_dataset(self):
        """ Generates a list of samples in the form (path_to_sample, target) or 
        (Tuple[path_to_sample], target) in the case of multimodal dataset.
        It considers the current 'self.split' to generate this list.

        Returns
        ----------
        samples: List[Union[Tuple[str, int], Tuple[List[str], int]]
            List of (path, target) or (List[path], target) in the case of multimodality.
        """

        participants = pd.read_csv(os.path.join(self.root, "participants.tsv"), sep="\t")
        
        if self.split == "train":
            # Training subset
            mask = participants.split.eq("train")
        elif self.split == "val":
            # Validation subset
            mask = participants.split.isin(["internal_test", "external_test"])
        elif self.split == "internal_val":
            # Subset of validation
            mask = participants.split.eq("internal_test")
        elif self.split == "external_val":
            # Another subset of validation
            mask = participants.split.eq("external_test")
        else:
            raise ValueError(f"Unkown split: {self.split}")

        participants = participants[mask]
        
        img_paths = {
                "vbm_roi": "sub-%i_preproc-cat12vbm_desc-gm_ROI.npy",
                "vbm": "sub-%i_preproc-cat12vbm_desc-gm_T1w.npy",
                "quasiraw": "sub-%i_preproc-quasiraw_T1w.npy",
                "fs_xhemi": "sub-%i_preproc-freesurfer_desc-xhemi_T1w.npy",
                "fs_desikan_roi": "sub-%i_preproc-freesurfer_desc-desikan_ROI.npy",
                "fs_destrieux_roi": "sub-%i_preproc-freesurfer_desc-destrieux_ROI.npy"}
    
        samples = []
        split = "train" if self.split == "train" else "val"
        target = "siteXacq" if self.target == "site" else self.target
        for (id, t) in participants[["participant_id", target]].values:
            sample = []
            for mod in self.modality:
                file_path = os.path.join(self.root, split, "derivatives", 
                                         f"sub-%i"%id, "ses-1", img_paths[mod] % id)
                # assert os.path.isfile(file_path)
                sample.append(file_path)
            if len(sample) == 1:
                samples.append((sample[0], t))
            else:
                samples.append((tuple(sample), t))
        return samples


    @staticmethod
    def fetch_dataset(root: str, verbose: bool=False):
        if not os.path.exists(root):
            if verbose:
                print(f"Cloning {OpenBHB.REPO_URL} in {os.path.dirname(root)}")
            if not shutil.which("git") or not shutil.which("git-lfs"):
                raise ValueError("Both git and git-lfs must be installed.")
            subprocess.run(f"git clone {OpenBHB.REPO_URL} {os.path.dirname(root)}", 
                           shell=True, check=True)
        else:
            if verbose:
                print(f"[INFO] Directory '{root}' already exists. Skipping clone.")
            return
        if verbose:
            print("[DONE] Dataset fetched successfully.")


    def get_neuromorphometric_atlas(self):
        nii = nibabel.load(os.path.join(self.root, "resource", "neuromorphometrics.nii"))
        labels = pd.read_csv(os.path.join(self.root, "resource", "neuromorphometrics.csv"), sep=";")
        return dict(data=nii, labels=list(labels["ROIabbr"].values))
    
    def get_cat12_template(self):
        nii = nibabel.load(os.path.join(self.root, "resource", "cat12vbm_space-MNI152_desc-gm_TPM.nii.gz"))
        return nii
    
    def get_quasiraw_template(self):
        nii = nibabel.load(os.path.join(self.root, "resource", "quasiraw_space-MNI152_desc-brain_T1w.nii.gz"))
        return nii

    def get_freesurfer_destrieux_labels(self, symmetric=False): # Get ROI names on Destrieux
        """ Return ROI names on Destrieux atlas (148 regions or 74 if symmetric=True)
            First 74 are left hemisphere, last 74 are right hemisphere
         
        Parameters
        ----------
        symmetric: bool
            If True, removes "lh-" and "rh-" from labels indicating right and left hemisphere.
            Final length is divided by two.
        """
        with open(os.path.join(self.root, "resource", "resources.json")) as f:
            labels = json.load(f)["destrieux_roi"]["features"]
        if symmetric:
            labels = [l[3:] for l in labels[:74]]
        return labels

    def get_freesurfer_desikan_labels(self, symmetric=False): # Get ROI names on Desikan
        """ Return ROI names on Desikan atlas (68 regions or 34 if symmetric=True)
            First 34 are left hemisphere, last 34 are right hemisphere

        Parameters
        ----------
        symmetric: bool
            If True, removes "lh-" and "rh-" from labels indicating right and left hemisphere.
            Final length is divided by two.
        """
        with open(os.path.join(self.root, "resource", "resources.json")) as f:
            labels = json.load(f)["desikan_roi"]["features"]
        if symmetric:
            labels = [l[3:] for l in labels[:34]]
        return labels
    
    def get_vbm_roi_labels(self): # Get ROI names on Neuromorphometric
        """ Return ROI names on Neuromorphometric atlas (142 regions for GM volumes, 142 regions for CSF volume)
            First 142 are GM volumes, last 142 are CSF volumes across the entire brain.
        """
        with open(os.path.join(self.root, "resource", "resources.json")) as f:
            labels = json.load(f)["vbm_roi"]["features"]
        # Remove "GM_Vol" and "CSF_Vol" that is not part of atlas name
        labels = [l.replace("_GM_Vol", "").replace("_CSF_Vol", "") for l in labels]
        return labels
    
    def get_freesurfer_channels(self):
        channels = pd.read_csv(os.path.join(self.root, "resource", "freesurfer_channels.txt"),
                               names=["channel"])
        return {v: k for (k, v) in channels["channel"].reset_index().values}
    
    def get_freesurfer_xhemi_channels(self):
        channels = pd.read_csv(os.path.join(self.root, "resource", "freesurfer_xhemi_channels.txt"), 
                               names=["channel"])
        return {v: k for (k, v) in channels["channel"].reset_index().values}


    def _get_shape(self):
        with open(os.path.join(self.root, "resource", "resources.json")) as f:
            r = json.load(f)
        preproc_map = {"vbm": "vbm", "quasiraw": "quasiraw", "vbm_roi": "vbm_roi", 
                       "fs_desikan_roi": "desikan_roi", "fs_destrieux_roi": "destrieux_roi", 
                       "fs_xhemi": "xhemi"}
        if len(self.modality) == 1:
            shape = r[preproc_map[self.modality[0]]]["shape"]
            shape[0] = len(self.samples)
            return shape
        else:
            shape = [r[preproc_map[mod]]["shape"] for mod in self.modality]
            for s in shape: s[0] = len(self.samples)
            return shape

    @property
    def shape(self):
        return self._shape

    def load_sample(self, sample):
        if isinstance(sample, str):
            # Returns a simple array
            return np.load(sample)[0]
        else:
            # Returns a dict {modality: array} 
            return {mod: np.load(s)[0] for (mod, s) in zip(self.modality, sample)}

    def __getitem__(self, idx: int):
        sample, target = self.samples[idx]
        sample = self.load_sample(sample)
        if self.transforms is not None:
            sample = self.transforms(sample)
        if self.target_transforms is not None:
            target = self.target_transforms(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def __str__(self):
        if len(self.modality) == 1:
            return f"openBHB_{self.modality[0]}_{self.split}_{self.target}"
        else:
            modalities = "-".join(self.modality)
            return f"openBHB_{modalities}_{self.split}_{self.target}"
