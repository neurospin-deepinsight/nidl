from torch.utils.data import Dataset
from typing import Union, Iterable, Callable
import numpy as np
import pandas as pd
import os
import shutil
import nibabel
import subprocess
from pathlib import Path
from collections import defaultdict

class OpenMind(Dataset):
    """
    OpenMind dataset [1] for self-supervised learning. It includes 3D brain images from 23 modalities (T1w, T2w, FLAIR,  Angio, etc.) 
    of 34k patients and 114k images (71k 3D MRI scans + 43k preprocessed 3D DWI images). Each image comes with a defacing mask and 
    an anatomy mask  (excluding non-brain tissues). It also comes with an image quality score for each image (1 is the best, 5 is the worse)
    and some partial meta-data (age, sex, BMI, race, weight, health status). 

    If not already present, data are downloaded automatically in the root directory (~1To).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv("openneuro_metadata.csv")
    >>> df['age'].describe()
    count           79761
    mean        28.180163
    std         17.845728
    min          2.000000
    25%         16.000000
    50%         23.000000
    75%         33.000000
    max        100.000000
    >>> set(df['modality'].values)
    {'MD', 'T2map', 'inplaneT2', 'T2w', 'UNIT1_denoised', 'T1w', 'angio', 'T1map', 'FA', 'SWI', 
    'sbref', 'T2starw', 'minIP', 'MP2RAGE', 'PET', 'ADC', 'UNIT1', 'T2starmap', 'FLASH', 'PDw', 
    'FLAIR', 'TRACE', 'inplaneT1', 'DWI'}
    >>> df['image_quality_score'].describe()
    count           113921
    mean          2.391286
    std           1.099961
    min           1.000000
    25%           1.400000
    50%           2.000000
    75%           3.000000
    max           5.000000

    [1] An OpenMind for 3D medical vision self-supervised learning, Wald et al., arXiv, 2025 
    """

    REPO_URL = "https://huggingface.co/datasets/AnonRes/OpenMind"

    def __init__(self, 
                 root: str, 
                 modality: Union[None, str, Iterable[str]]=None,
                 target: str="age",
                 filter_na_target: bool=False,
                 transforms: Callable=None,
                 target_transforms: Callable=None,
                 verbose: bool=False): 
        """
        Parameters
        ----------
        root: str
            Path to the root data directory containing "OpenMind" directory fetched from HuggingFace.

        modality: str or Iterable[str] in {'MD', 'T2map', 'inplaneT2', 'T2w', 'UNIT1_denoised', 'T1w', 'angio', 
                'T1map', 'FA', 'SWI', 'sbref', 'T2starw', 'minIP', 'MP2RAGE', 'PET', 'ADC', 'UNIT1', 
                'T2starmap', 'FLASH', 'PDw', 'FLAIR', 'TRACE', 'inplaneT1', 'DWI'}

            Which modalities to load for each brain image. When 'modality' is a tuple, 
            a dict is generated in __getitem__ where keys are the modality names 
            and values are a tuple of 3 nibabel.nifti1.Nifti1Image object:
            
            {
                <modality_1>: (<image.nii>, <anatomical_mask.nii>, <defacing_mask.nii>),
                <modality_2>: ...
                ...
            }
            Some modalities may be None but at least one must be present per sample. 
            If a single modality is given, only a tuple (<image.nii>, <anatomical_mask.nii>, <defacing_mask.nii>) 
            is returned by __getitem__. If None, all modalities are included.

        target: str  in {'age', 'sex', 'bmi', 'race', 'weight', 'health_status'}
            Target value to return with each sample (multimodal or unimodal). 
            None values can be returned if 'filter_na_target' is False (default).

        filter_na_target: bool, default=False
            If True, removes samples with no available target values. 

        transforms: callable or None, default=None
            A function/transform that takes in a brain image and returns a transformed version.
            Exact input depends on "modality" (can be 3D image, 1D vector or a dictionary)

        target_transforms: callable or None, default=None
            A function/transform that takes in the target and returns a transformed version.

        verbose: bool, default=False
            If True, print checks and downloading steps if required.
        """
        
        valid_modalities = {'MD', 'T2map', 'inplaneT2', 'T2w', 'UNIT1_denoised', 'T1w', 'angio', 'T1map', 'FA', 'SWI', 
                            'sbref', 'T2starw', 'minIP', 'MP2RAGE', 'PET', 'ADC', 'UNIT1', 'T2starmap', 'FLASH', 'PDw', 
                            'FLAIR', 'TRACE', 'inplaneT1', 'DWI'}
        valid_targets = {'age', 'sex', 'bmi', 'race', 'weight', 'health_status'}

        root = os.path.expanduser(root)
        if modality is None:
            modality = tuple(sorted(list(valid_modalities)))
        elif isinstance(modality, str):
            modality = (modality,)
        for mod in modality:
            if mod not in valid_modalities:
                raise ValueError(f"'modality' must be in {valid_modalities}")
        
        if target not in valid_targets:
            raise ValueError(f"'target' must be in {valid_targets}")
        
        self.modality = modality
        self.root = root
        self.target = target
        self.filter_na_target = filter_na_target
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.fetch_dataset(root, verbose=verbose)
        self.samples = self.make_dataset(os.path.join(root, "OpenMind"), verbose=verbose)


    @staticmethod
    def fetch_dataset(root: str, verbose: bool=False):
        clone_dir = os.path.join(root, "OpenMind")
        if not os.path.exists(clone_dir):
            if verbose:
                print(f"Cloning {OpenMind.REPO_URL} in {root}")
            if not shutil.which("git") or not shutil.which("git-lfs"):
                raise ValueError("Both git and git-lfs must be installed.")
            subprocess.run(f"git clone {OpenMind.REPO_URL} {root}", shell=True, check=True)
        else:
            if verbose:
                print(f"[INFO] Directory '{clone_dir}' already exists. Skipping clone.")
            return
        if verbose:
            print("[DONE] Dataset fetched successfully.")


    def make_dataset(self, directory, verbose=False):
        """Builds a list of samples, one for each (subject, session) pair including a tuple of modalities 
        (as defined by 'self.modality') and a target. 
        
        Returns a list of (unique_id, [modality-aligned paths], target)
        
        """
        participants = pd.read_csv(os.path.join(directory, "openneuro_metadata.csv"), sep=',')
            
        grouped = defaultdict(lambda: {'paths': {}, 'metadata': {}, 'target': None})

        # Gather all requested modalities for each (subject, session) pair
        # If multiple runs are available for the same modality, choose the one 
        # with the highest image quality score.
        columns = ["image_path", "anat_mask_path", "anon_mask_path", "modality", "image_quality_score", self.target]
        img_rm = 0 # How many images were removed because of too low IQS and other runs available.
        for (p, anat_m, anon_m, m, iqs, t) in participants[columns].values:
            if self.filter_na_target and t is None: # filter by target
                continue # skip
            if m in set(self.modality): # filter by modality
                path = Path(p)
                parts = path.parts

                dataset = parts[0]  # e.g., 'ds000001'
                subject = next((part for part in parts if part.startswith("sub-")), None)
                session = next((part for part in parts if part.startswith("ses-")), "ses-DEFAULT")

                if not subject:
                    print(f"Warning: participant ID in {p} not determined")
                    continue  # skip if subject can't be determined

                unique_id = f"{dataset}__{subject}__{session}"
                
                if unique_id in grouped and m in grouped[unique_id]['metadata']:
                    img_rm += 1 # we will delete one image
                    old_iqs = grouped[unique_id]['metadata'][m]['image_quality_score']
                    if old_iqs >= iqs:
                        continue # skip the current image with low IQS
                    
                grouped[unique_id]['paths'][m] = (os.path.join(directory, "OpenMind", path), 
                                                  os.path.join(directory,  "OpenMind", anat_m), 
                                                  os.path.join(directory,  "OpenMind", anon_m))
                grouped[unique_id]['metadata'][m] = {
                    'image_quality_score': iqs
                }
                grouped[unique_id]["target"] = t # independent of modality
        if verbose:
            print(f"[INFO] {img_rm} images found with multiple runs and removed based on IQS")
        uids = sorted(list(grouped.keys()))

        if verbose:
            print(f"[INFO] {len(uids)} samples with unique (subject, session) ID found in {directory}")

        # Return structured list
        return [(grouped[uid]['paths'], grouped[uid]['target']) for uid in uids]
    

    def load_sample(self, sample):
        
        if len(self.modality) == 1:
            nii_path, anat_path, anon_path = sample[self.modality[0]]
            return (nibabel.load(nii_path), nibabel.load(anat_path), nibabel.load(anon_path))
        
        elif len(self.modality) > 1:
            result = dict()
            for m in self.modality:
                nii_path, anat_path, anon_path = sample.get(m, (None, None, None))
                if nii_path is not None:
                    result[m] = (nibabel.load(nii_path), nibabel.load(anat_path), nibabel.load(anon_path))
                else:
                    result[m] = (None, None, None)
            return result

        return None


    def __getitem__(self, idx: int):
        """ Returns a tuple (sample, target) for a given idx where:
        -   sample is either a tuple (<image.nii>, <anatomical_mask.nii>, <defacing_mask.nii>) (single-modality case)
            or a dict {<modality_1>: (<image.nii>, <anatomical_mask.nii>, <defacing_mask.nii>), 
                       <modality_2>: ...} (multi-modality case)
        -   target is float or str
        """
        sample, target = self.samples[idx]
        sample = self.load_sample(sample)
        if self.transforms is not None:
            sample = self.transforms(sample)
        if self.target_transforms is not None:
            target = self.target_transforms(target)
        return sample, target

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    openmind = OpenMind("/neurospin/psy/openmind", modality=["T1w", "T2w"], verbose=True)

    print(openmind[0])