import os
from typing import Any, Callable, Optional, Union

import nibabel
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset


class OpenBHB(Dataset):
    """OpenBHB dataset [1]_.

    The Open Big Healthy Brains (OpenBHB) dataset is a large multi-site brain
    MRI dataset consisting of 3227 training samples and 757 validation samples.
    It aggregates T1-weighted (T1w) MRI scans from 10 public datasets:

    - IXI
    - ABIDE I
    - ABIDE II
    - CoRR
    - GSP
    - Localizer
    - MPI-Leipzig
    - NAR
    - NPC
    - RBP

    These scans were acquired across 93 centers worldwide (North America,
    Europe, and China). Only healthy controls aged between 6 and 88 years
    are included, with balanced representation of males and females.

    All T1w MRI scans have been uniformly preprocessed using CAT12 (SPM),
    FreeSurfer, and Quasi-Raw (in-house minimal preprocessing). Both
    Voxel-Based Morphometry (VBM) and Surface-Based Morphometry (SBM) features
    are available.

    .. warning::
        The entire OpenBHB takes ~350GB of disk. We recommend enabling
        `streaming=True` if you intend to use only a small portion of the
        dataset.


    Parameters
    ----------
    root : str
        Path to the root data directory where the dataset is stored.

    modality : str or tuple of str
        Which modality to load for each brain image. If a tuple (multimodal
        OpenBHB), a dictionary is returned from `__getitem__` with modality
        names as keys and corresponding NumPy arrays as values.

        Available modalities:

        - "vbm": Whole-brain voxel-based morphometry 3D T1w image,
          shape `(121, 145, 121)`
        - "quasiraw": Whole-brain T1w image with minimal preprocessing,
          shape `(182, 218, 182)`
        - "vbm_roi": Gray matter volume per region (Neuromorphometrics atlas,
          142 regions by hemisphere), shape `(1, 284)`
        - "fs_desikan_roi": FreeSurfer surface-based features computed on the
          Desikan atlas (34 regions by hemisphere), shape `(7, 68)`
        - "fs_destrieux_roi": FreeSurfer surface-based features computed on the
          Destrieux atlas (74 regions by hemisphere), shape `(7, 148)`
        - "fs_xhemi": FreeSurfer surface-based features (curvature, sulcal
          depth, cortical thickness) computed on the `fsaverage7` mesh
          (163842 vertices by hemisphere), shape `(8, 163842)`

    target : {'age', 'sex', 'site'}, list of str, or None
        Target(s) to return with each image. If string, returns the target as
        float (for 'age'), int (for 'site') or string (for 'sex'). If `target`
        is a list of strings, returns multiple targets as dictionary:
        {<target>: <value>}. If None, no target is returned.

    split : {'train', 'val', 'internal_val', 'external_val'}
        Dataset split to use. The 'val' split is the union of:

        - 'internal_val': Images acquired with the same MRI scanner as training
          data (in-domain)
        - 'external_val': Images acquired with different MRI scanners
          (out-of-domain)

    streaming : bool, default=True
        If True, data are downloaded lazily from Hugging Face on demand (when
        accessed via `__getitem__`). If False, the entire split is downloaded
        at initialization for the requested modality.

    max_workers: int, default=1
        Number of concurrent threads to download files on the Hugging Face,
        1 thread = 1 file download.
        Warning: setting `max_workers` > 1 can raise Hugging Face 429 errors
        (too many requests). We recommend keeping this value low.

    transforms : callable or None, default=None
        A function/transform that takes in a brain image and returns a
        transformed version. Input depends on `modality` and can be a 3D image,
        1D vector, or dict.

    target_transforms : callable or None, default=None
        A function/transform applied to the target(s).


    Examples
    ----------
    Load the VBM modality from the training split and get the age target:

    >>> dataset = OpenBHB(
    ...    root='data/openbhb', modality='vbm', target='age',
    ...    split='train'
    ... )
    >>> image, age = dataset[0]
    >>> print(image.shape)
    (1, 121, 145, 121)
    >>> print(age)
    34.0

    Load multiple modalities and multiple targets:

    >>> dataset = OpenBHB(
    ...     root='data/openbhb',
    ...     modality=('vbm', 'quasiraw'),
    ...     target=['age', 'sex', 'site'],
    ...     split='val'
    ... )
    >>> data, targets = dataset[10]
    >>> print(data['vbm'].shape)
    (1, 121, 145, 121)
    >>> print(data['quasiraw'].shape)
    (1, 182, 218, 182)
    >>> print(targets)
    {'age': 19.0, 'sex': 'female', 'site': 0}


    Notes
    ----------
    The data are downloaded exclusively from the `OpenBHB repository
    <https://huggingface.co/datasets/benoit-dufumier/openBHB>`_ in the
    HuggingFace either on-the-fly (lazy download) or during initialization
    (immediate download) if there are not already there.


    References
    ----------
    .. [1] Dufumier, B., Grigis, A., Victor, J., Ambroise, C., Frouin, V. &
           Duchesnay, E. (2022). OpenBHB: a Large-Scale Multi-Site Brain MRI
           Data-set for Age Prediction and Debiasing.
           NeuroImage, 254, 119121. https://doi.org/10.1016/j.neuroimage.2022.119637

    """

    REPO_ID = "benoit-dufumier/openBHB"
    REVISION = "8508cda68fea74f217926acbf46ee5863f8879d1"  # commit ID

    def __init__(
        self,
        root: str,
        modality: Union[str, tuple[str, ...]] = "vbm",
        target: Union[str, list[str], None] = "age",
        split: str = "train",
        streaming: bool = True,
        max_workers: int = 1,
        transforms: Optional[Callable] = None,
        target_transforms: Optional[Callable] = None,
    ):
        self.root = self._parse_root(root)
        self.modality = self._parse_modality(modality)
        self.target = self._parse_target(target)
        self.split = self._parse_split(split)
        self.streaming = streaming
        self.transforms = transforms
        self.target_transforms = target_transforms

        self.samples = self.make_dataset(self.split)
        if not self.streaming:  # fetch all data split if not there
            self.download_dataset_split(
                split=self.split,
                modality=self.modality,
                samples=self.samples,
                incremental=True,
                max_workers=max_workers,
            )
        self._cache = {}

    def download_file(self, filename: str) -> str:
        """Download a single file from the OpenBHB repository on the HF."""
        try:
            from huggingface_hub import hf_hub_download
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"File '{filename}' cannot be downloaded from Hugging Face "
                "because the 'huggingface_hub' package is not installed. "
                "Please run 'pip install huggingface_hub' first."
            ) from e

        return hf_hub_download(
            repo_id=self.REPO_ID,
            revision=self.REVISION,
            filename=filename,
            repo_type="dataset",
            local_dir=self.root,
        )

    def download_dataset_split(
        self,
        split: str,
        modality: tuple[str, ...],
        samples: list[tuple[Any, Any]],
        incremental: bool = True,
        max_workers: int = 8,
    ):
        """Fetch a split of the dataset from Hugging Face if not present.

        Parameters
        ----------
        split: {'train', 'val', 'internal_val', 'external_val'}
            Split to download if not present.

        modality : tuple of str
            Modalities to download ("vbm", "vbm_roi", "quasiraw", "fs_xhemi",
            "fs_desikan_roi" or "fs_destrieux_roi")

        samples: list of dict
            List of paths to the data in the current split. This should have
            been generated by `make_dataset`.

        incremental: bool, default=True
            If True, only missing files in the data split are downloaded.
            Otherwise, all data in the split are downloaded and local data
            are eventually replaced.

        max_workers: int, default=8
            Number of concurrent threads to download files (1 thread = 1 file
            download).
        """
        split = "train" if split == "train" else "val"
        img_regex = {
            "vbm": f"{split}/derivatives/sub-*/ses-*/sub-*cat12vbm_desc-gm_T1w.npy",
            "quasiraw": f"{split}/derivatives/sub-*/ses-*/sub-*quasiraw_T1w.npy",
            "fs_xhemi": f"{split}/derivatives/sub-*/ses-*/sub-*xhemi_T1w.npy",
            "vbm_roi": f"{split}/derivatives/cat12vbm_roi/cat12vbm_roi_features.csv",
            "fs_desikan_roi": f"{split}/derivatives/freesurfer_roi/desikan_roi_features.csv",
            "fs_destrieux_roi": f"{split}/derivatives/freesurfer_roi/destrieux_roi_features.csv"
        }
        allow_patterns = [
            img_regex[m] for m in modality
        ]
        ignore_patterns = []
        len_all = 0
        if incremental:
            # Ignore already downloaded files.
            for paths, _ in samples:
                paths = next(iter(paths.values()))
                if isinstance(paths, str):
                    paths = (paths,)
                for p in paths:
                    len_all += 1
                    if os.path.isfile(p):
                        ignore_patterns.append(p)

        if not incremental or len(ignore_patterns) < len_all:
            try:
                from huggingface_hub import snapshot_download
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    f"Split '{split}' cannot be downloaded from Hugging Face "
                    "because the 'huggingface_hub' package is not installed. "
                    "Please run 'pip install huggingface_hub' first."
                ) from e
            snapshot_download(
                repo_id=self.REPO_ID,
                revision=self.REVISION,
                repo_type="dataset",
                local_dir=self.root,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                max_workers=max_workers,
            )

    def get_or_download_file(self, path: str) -> str:
        if not os.path.isfile(path):
            # Download the file from HF
            self.download_file(path.replace(self.root + "/", ""))
        return path

    def get_participants(self):
        path = os.path.join(self.root, "participants.tsv")
        return pd.read_csv(self.get_or_download_file(path), sep="\t")

    def get_resource(self):
        path = os.path.join(self.root, "resource", "resources.json")
        return pd.read_json(self.get_or_download_file(path))

    def make_dataset(self, split: str):
        """Generate a list of sample file paths and their corresponding targets
        for a given dataset split.

        This method constructs file paths for each participant listed in
        `participants.tsv` according to the specified `split`. It supports
        both unimodal and multimodal configurations, depending on the
        `modality` attribute.

        Each returned sample is a tuple of the form:

        - ({id: path}, target): if `modality` is a single string
        - ({id: tuple of path}, target): if `modality` is a tuple of strings

        If `target` is None, the sample tuple excludes the target and only
        contains the path or tuple of paths.

        Parameters
        ----------
        split: {'train', 'val', 'internal_val', 'external_val'}
            Which participants to include in the dataset.

        Returns
        ----------
        samples: list of tuple
            List of samples in the form ({id: path}, target), where:

            - `id` is the participant's ID.
            - `path` is either a single file path (str) or a tuple of file
              paths (if multiple modalities are used).
            - `target` is the associated value from the participants metadata
              (e.g., age, sex, site), or None if `target` is not set.

        Notes
        ----------
        Expects the following file under the root directory:
        `<root>/participants.tsv`. If not present, it is downloaded
        automatically from the Hugging Face.
        """

        participants = self.get_participants()

        if split == "train":
            # Training subset
            mask = participants.split.eq("train")
        elif split == "val":
            # Validation subset
            mask = participants.split.isin(["internal_test", "external_test"])
        elif split == "internal_val":
            # Subset of validation
            mask = participants.split.eq("internal_test")
        elif split == "external_val":
            # Another subset of validation
            mask = participants.split.eq("external_test")
        else:
            raise ValueError(f"Unkown split: {split}")

        participants = participants[mask]

        img_paths = {
            "vbm_roi": "derivatives/cat12vbm_roi/cat12vbm_roi_features.csv",
            "vbm": (
                "derivatives/sub-{id}/ses-1/"
                "sub-{id}_preproc-cat12vbm_desc-gm_T1w.npy"
            ),
            "quasiraw": (
                "derivatives/sub-{id}/ses-1/sub-{id}_preproc-quasiraw_T1w.npy"
            ),
            "fs_xhemi": (
                "derivatives/sub-{id}/ses-1/"
                "sub-{id}_preproc-freesurfer_desc-xhemi_T1w.npy"
            ),
            "fs_desikan_roi": (
                "derivatives/freesurfer_roi/desikan_roi_features.csv"
            ),
            "fs_destrieux_roi": (
                "derivatives/freesurfer_roi/destrieux_roi_features.csv"
            ),
        }

        samples = []
        split = "train" if self.split == "train" else "val"

        # Determine target columns and convert to NumPy
        target_cols = []
        if self.target is not None:
            if isinstance(self.target, str):
                target_cols = [self.target]
            elif isinstance(self.target, (list, tuple)):
                target_cols = list(self.target)
            else:
                raise ValueError(f"Invalid target type: {self.target}")

            if "site" in target_cols:
                participants["site"] = participants["siteXacq"].astype(int)

        data = participants[["participant_id", *target_cols]].to_dict(
            orient="records"
        )

        for entry in data:
            id = int(entry["participant_id"])
            sample = {id: []}
            target = None
            if isinstance(self.target, str):
                target = entry[target_cols[0]]
            else:
                target = {k: entry[k] for k in target_cols}
            for mod in self.modality:
                file_path = os.path.join(
                    self.root,
                    split,
                    img_paths[mod].format(id=id),
                )
                sample[id].append(file_path)
            if len(sample[id]) == 1:  # one modality
                samples.append(({id: sample[id][0]}, target))
            else:  # multi-modal openBHB
                samples.append(({id: tuple(sample[id])}, target))
        return samples

    def get_neuromorphometrics_atlas(self):
        """Get the Neuromorphometrics gray matter atlas and its region names.

        This method loads the Neuromorphometrics gray matter atlas as a NIfTI
        image, along with the associated region names (abbreviations).

        Returns
        ----------
        dict
            A dictionary containing:

            - `data` : :class:`nibabel.Nifti1Image`, the atlas image.
            - `labels` : list of region names (string) corresponding to
              integer labels in the atlas.

        Notes
        ----------
        Expects the following files under the resource directory:

        - `<root>/resource/neuromorphometrics.nii` : NIfTI atlas file
        - `<root>/resource/neuromorphometrics.csv` : CSV with region names.

        If the files are not found locally, they will be downloaded from the
        Hugging Face.

        See Also
        ----------
        nibabel.load : Function used to load the NIfTI image.
        """
        niipath = self.get_or_download_file(
            os.path.join(self.root, "resource", "neuromorphometrics.nii")
        )
        labelpath = self.get_or_download_file(
            os.path.join(self.root, "resource", "neuromorphometrics.csv")
        )
        nii = nibabel.load(niipath)
        labels = pd.read_csv(labelpath, sep=";")
        return {"data": nii, "labels": list(labels["ROIabbr"].values)}

    def get_cat12_template(self):
        """Get the CAT12 gray matter tissue probability map as NIfTI image.

        This method retrieves the CAT12 gray matter (GM) tissue probability map
        (TPM) registered to MNI152 space.

        Returns
        ----------
        nii : nibabel.Nifti1Image, shape (121, 145, 121)
            A 3D NIfTI image containing the CAT12 gray matter TPM.

        Notes
        ----------
        The template file is expected at:
        `<root>/resource/cat12vbm_space-MNI152_desc-gm_TPM.nii.gz`. If the file
        is not available locally, it will be downloaded from the Hugging Face.

        See Also
        ----------
        nibabel.load : Function used to load the NIfTI image.
        """
        nii = nibabel.load(
            self.get_or_download_file(
                os.path.join(
                    self.root,
                    "resource",
                    "cat12vbm_space-MNI152_desc-gm_TPM.nii.gz",
                )
            )
        )
        return nii

    def get_quasiraw_template(self):
        """Get the quasi-raw MNI152 brain template as a NIfTI image.

        This method retrieves the quasi-raw T1-weighted brain template in
        MNI152 space.

        Returns
        ----------
        nii : nibabel.Nifti1Image, shape (182, 218, 182)
            A 3D NIfTI image containing the quasi-raw MNI152 brain template.

        Notes
        ----------
        The template file is expected at:
        `<root>/resource/quasiraw_space-MNI152_desc-brain_T1w.nii.gz`. If the
        file is not present locally, it is automatically downloaded from the
        Hugging Face.

        See Also
        ----------
        nibabel.load : Function used to load the NIfTI image.
        """
        nii = nibabel.load(
            self.get_or_download_file(
                os.path.join(
                    self.root,
                    "resource",
                    "quasiraw_space-MNI152_desc-brain_T1w.nii.gz",
                )
            )
        )
        return nii

    def get_fs_labels(self, atlas: str = "destrieux", symmetric: bool = False):
        """Get region names on the given atlas where "fs_destrieux_roi" (for
        "destrieux" atlas) or "fs_desikan_roi" (for "desikan" atlas) features
        have been computed in OpenBHB.

        The names are extracted from the resource file.

        First 74 (resp. 38) regions are from the left hemisphere, last 74
        (resp. 38) are from the right hemisphere for the Destrieux
        (resp. Desikan).

        Parameters
        ----------
        symmetric: bool
            If True, removes "lh-" and "rh-" from labels indicating right and
            left hemisphere. Final length is divided by two.

        Returns
        ----------
        labels: list of string
            List of region names on the given atlas.

        Notes
        ----------
        The resource file is expected at: `<root>/resource/resources.json`. If
        it is not present locally, it is automatically downloaded from the
        Hugging Face.
        """
        if atlas not in ["destrieux", "desikan"]:
            raise ValueError(
                f"`atlas` should be in 'destrieux' or 'desikan', got {atlas}"
            )
        n_regions = 74 if atlas == "destrieux" else 34
        resource = self.get_resource()
        labels = resource[f"{atlas}_roi"]["features"]
        if symmetric:
            labels = [
                label[3:] for label in labels[:n_regions]
            ]  # rm 'lh-' or 'rh-'
        return labels

    def get_vbm_roi_labels(self):
        """Get region names on the Neuromorphometrics atlas where "vbm_roi"
        features are computed in OpenBHB.

        The names are extracted from the resource file. If it is not present
        locally, it is automatically downloaded from the Hugging Face.

        First 142 features are GM volumes, last 142 are CSF volumes.

        Returns
        ----------
        labels: list of string
            List of region names on the Neuromorphometrics atlas where
            "vbm_roi" features have been computed.

        Notes
        ----------
        The resource file is expected at: `<root>/resource/resources.json`
        """
        resource = self.get_resource()
        return resource["vbm_roi"]["features"]

    def get_fs_roi_feature_names(self):
        """Get the 7 feature names corresponding to "fs_destrieux_roi" and
        "fs_desikan_roi" data.

        The feature names are extracted from the resource file.

        Returns
        ----------
        features: list of string
            List of 7 feature names corresponding to "fs_destrieux_roi" and
            "fs_desikan_roi" data in OpenBHB.

        Notes
        ----------
        The resource file is expected at: `<root>/resource/resources.json`. If
        it is not present locally, it is automatically downloaded from the
        Hugging Face.

        """
        resource = self.get_resource()
        return resource["destrieux_roi"]["channels"]

    def get_fs_xhemi_feature_names(self):
        """Get the 8 feature names corresponding to "fs_xhemi" data.

        The feature names are extracted from the resource file. If it is not
        present locally, it is automatically downloaded from the Hugging Face.

        Returns
        ----------
        features: list of string
            List of 8 feature names corresponding to "fs_xhemi".

        Notes
        ----------
        The resource file is expected at: `<root>/resource/resources.json`
        """
        resource = self.get_resource()
        return resource["xhemi"]["channels"]

    def __getitem__(self, idx: int):
        """Retrieve one sample (and optionally its target) from the dataset.

        This method loads the sample located at the given index, applies any
        specified input and target transformations, and returns the processed
        data. If the dataset is configured with multiple modalities, the sample
        will be a dictionary of arrays.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        ----------
        sample : array or dict of array
            The loaded sample. If unimodal, this is a single NumPy array of
            shape:

            - `(1, 121, 145, 121)` if  `modality == "vbm"`
            - `(1, 182, 218, 182)` if `modality == "quasiraw"`
            - `(1, 284)` if `modality == "vbm_roi"`
            - `(7, 68)` if `modality == "fs_desikan_roi"`
            - `(7, 148)` if `modality == "fs_destrieux_roi"`
            - `(8, 163842)` if `modality == "fs_xhemi"`

        target : optional
            The target associated with the sample, after applying
            `self.target_transforms`, if provided. This is only returned if
            `self.target` is not None.
            If `self.target` is a single string ("age", "sex", or "site"), the
            target is returned as a single value.
            If `self.target` is a list or tuple of strings, the target is
            returned as a dictionary mapping each target name to its value:
            `{<target>: <value>}`.

        Notes
        ----------
        - If `self.target` is None, only the `sample` is returned.
        - If `self.transforms` or `self.target_transforms` are specified, they
          are applied to the sample and target respectively.
        - The sample path and target are retrieved from `self.samples`, which
          should be constructed by `make_dataset`.
        """
        sample_path, target = self.samples[idx]
        sample = self._load_sample(sample_path)
        if self.transforms is not None:
            sample = self.transforms(sample)
        if self.target_transforms is not None:
            target = self.target_transforms(target)
        if self.target is None:
            return sample
        else:
            return sample, target

    def _load_sample(self, sample_path):
        id, sample_path = next(iter(sample_path.items()))
        if isinstance(sample_path, str):
            # Returns a simple array
            sample_path = self.get_or_download_file(sample_path)
            return self._load_data(id, sample_path)
        else:
            # Returns a dict {<modality>: array}
            for s in sample_path:
                self.get_or_download_file(s)
            return {
                mod: self._load_data(id, s)
                for (mod, s) in zip(self.modality, sample_path)
            }

    def _load_data(self, id: int, path: str):
        if path.endswith(".csv"):
            if path.endswith("cat12vbm_roi_features.csv"):
                shape = (1, 284)
            elif path.endswith("desikan_roi_features.csv"):
                shape = (7, 68)
            elif path.endswith("destrieux_roi_features.csv"):
                shape = (7, 148)
            if path in self._cache:
                    data = self._cache[path]
            else:
                data = pd.read_csv(path)
                self._cache[path] = data
            row = data[data.participant_id.astype(int).eq(id)]
            row = row.drop(columns=["participant_id", "session"]).to_numpy()
            return row.reshape(shape).astype(np.float32)
        else:
            return np.load(path)[0].astype(np.float32)

    def _parse_root(self, path):
        # eventually parse "~" or $HOME
        path = os.path.expanduser(path)
        # check if exists, otherwise create it
        if not os.path.isdir(path):
            os.makedirs(path)
        return path

    def _parse_modality(self, modality):
        valid_modalities = [
            "vbm",
            "vbm_roi",
            "quasiraw",
            "fs_desikan_roi",
            "fs_destrieux_roi",
            "fs_xhemi",
        ]
        if isinstance(modality, str):
            modality = (modality,)
        elif isinstance(modality, (tuple, list)):
            modality = tuple(modality)
        else:
            raise ValueError(
                f"`modality` must be str or tuple of str, got {modality}"
            )
        if len(modality) == 0:
            raise ValueError("`modality` cannot be an empty tuple")
        for mod in modality:
            if mod not in valid_modalities:
                raise ValueError(
                    f"`modality` must be in {valid_modalities}, got {modality}"
                )
        return modality

    def _parse_split(self, split):
        valid_splits = ["train", "val", "internal_val", "external_val"]
        if split not in valid_splits:
            raise ValueError(f"`split` must be in {valid_splits}, got {split}")
        return split

    def _parse_target(self, target):
        valid_targets = ["age", "sex", "site"]
        if target is None:
            return target
        elif isinstance(target, str):
            if target not in valid_targets:
                raise ValueError(
                    f"`target` must be str in {valid_targets}, got {target}"
                )
            return target
        if not isinstance(target, (tuple, list)):
            raise ValueError(
                "`target` must be str, tuple of str or None, "
                f"got {type(target)}"
            )
        if len(target) == 0:
            raise ValueError("`target` must be a non-empty list or None.")
        for t in target:
            if t not in valid_targets:
                raise ValueError(
                    f"`target` must be in {valid_targets}, got {t}"
                )
        return tuple(target)

    def __len__(self):
        return len(self.samples)

    def __str__(self):
        if len(self.modality) == 1:
            return f"openBHB_{self.modality[0]}_{self.split}_{self.target}"
        else:
            modalities = "-".join(self.modality)
            return f"openBHB_{modalities}_{self.split}_{self.target}"
