from torch.utils.data import Dataset
import os
import glob
import errno
import nibabel
import numpy as np
import pandas as pd

class BidsDataset(Dataset):
    """ A scalable neuroimaging dataset organized with BIDS.

    Parameters
    ----------
    root: str
        the location where are stored the data.

    patterns: str or list of str
        the regex that can be used to retrieve the images of interest or any
        data that can be retrieved by nibabel.load.

    subject_in_patterns: int or list of int
        the folder level where the subject identifiers can be retrieved.

    targets: str or list of str
        the dataset will also return these tabular data. A 'participants.tsv'
        file containing subject information (including the requested targets)
        is expected at the root.
        
    target_mapping: dict, default None
        optionaly, define a dictionary specifying different replacement values
        for different existing values. See pandas DataFrame.replace
        documentation for more information.

    split: str, default 'train'
        define the split to be considered. A '<split>.tsv' file containg the
        subject to include us expected at the root.

    transforms: callable or list of callable, default None
        a function that can be called to augment the input images.

    mask: array, default None
        optionnaly, mask the input image.

    contrastive: bool, default False
        optionaly, create a contrastive dataset that will return a pair of
        augmented images.

    n_max: int, default None
        optionaly, keep only a subset of subjects (for debuging purposes).

    withdraw_subjects: list of str, default None
        optionaly, provide a list of subjects to remove from the dataset.
    """
    def __init__(self, root, patterns, subject_in_patterns, targets,
                 target_mapping=None, split="train", transforms=None,
                 mask=None, contrastive=False, n_max=None,
                 withdraw_subjects=None):

        # Sanity
        if not isinstance(patterns, (list, tuple)):
            patterns = [patterns]
        if not isinstance(subject_in_patterns, (list, tuple)):
            subject_in_patterns = [subject_in_patterns]
        if not isinstance(targets, (list, tuple)):
            targets = [targets]
        if not isinstance(transforms, (list, tuple)):
            transforms = [transforms] * len(patterns)
        assert len(patterns) == len(transforms)
        assert len(patterns) == len(subject_in_patterns)
        participant_file = os.path.join(root, "participants.tsv")
        split_file = os.path.join(root, f"{split}.tsv")
        for path in (participant_file, split_file):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), path)

        # Parameters
        self.root = root
        self.patterns = patterns
        self.n_modalities = len(patterns)
        self.targets = targets
        self.target_mapping = target_mapping
        self.split = split
        self.transforms = transforms
        self.mask = mask
        self.contrastive = contrastive

        # Load subjects / data location
        self.info_df = pd.read_csv(participant_file, sep="\t")
        self.info_df = self.info_df.astype({"participant_id": "str"})
        self.split_df = pd.read_csv(split_file, sep="\t")
        self.split_df = self.split_df[["participant_id"]]
        self.split_df = self.split_df.astype({"participant_id": "str"})
        if withdraw_subjects is not None:
            self.split_df = self.split_df[
                ~self.split_df["participant_id"].isin(withdraw_subjects)]
        self._df = pd.merge(self.split_df, self.info_df, on="participant_id")
        self.mod_names = []
        for idx, pattern in enumerate(patterns):
            _regex = os.path.join(root, pattern)
            _sidx = subject_in_patterns[idx]
            _files = dict((path.split(os.sep)[_sidx].replace("sub-", ""), path)
                           for path in glob.glob(_regex))
            self._df[f"data{idx}"] = [
                _files.get(subject) for subject in self._df["participant_id"]]
            self.mod_names.append(f"data{idx}")

        # Keep only useful information / sanitize
        self._df = self._df[self.mod_names + targets]
        _missing_data = self._df[self._df.isnull().any(axis=1)]
        if len(_missing_data) > 0:
            print(_missing_data)
            raise ValueError(f"Missing data in {split}!")
        self._df.replace(target_mapping, inplace=True)
        if n_max is not None and len(self._df) > n_max:
            self._df = self._df.head(n_max)
        self.data = self._df[self.mod_names].values
        self.target = self._df[targets].values

    def __repr__(self):
        return (f"{self.__class__.__name__}<split='{self.split}',"
                f"modalities={self.n_modalities},targets={self.targets},"
                f"contrastive={self.contrastive}>")

    def __getitem__(self, idx):
        data = []
        for path, trf in zip(self.data[idx], self.transforms):
            arr = nibabel.load(path).get_fdata().astype(np.float32)
            if self.mask is not None:
                arr[np.where(self.mask == 0)] = 0
            arr = np.expand_dims(arr, axis=0)
            if self.contrastive:
                assert trf is not None
                arr = np.stack((trf(arr), trf(arr)), axis=0)
            elif trf is not None:
                arr = trf(arr)
            data.append(arr)
        return *data, *self.target[idx]

    def __len__(self):
        return len(self._df)