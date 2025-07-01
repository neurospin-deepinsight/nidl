##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


""" Base class to generate datasets.
"""

import abc
import errno
import glob
import hashlib
import os
import warnings

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """ Base neuroimaging dataset.

    Notes
    -----
    A 'participants.tsv' file containing subject information (including the
    requested targets) is expected at the root.
    A '<split>.tsv' file containg the subject to include is expected at the
    root.

    Parameters
    ----------
    root: str
        the location where are stored the data.
    patterns: str or list of str
        the relative locations of your data.
    channels: str or list of str, default=None
        the name of the channels.
    split: str, default 'train'
        define the split to be considered.
    targets: str or list of str, default=None
        the dataset will also return these tabular data.
    target_mapping: dict, default None
        optionaly, define a dictionary specifying different replacement values
        for different existing values. See pandas DataFrame.replace
        documentation for more information.
    transforms: callable, default None
        a function that can be called to augment the input images.
    mask: str, default None
        optionnaly, mask the input data using this numpy array.
    withdraw_subjects: list of str, default None
        optionaly, provide a list of subjects to remove from the dataset.

    Raises
    ------
    FileNotFoundError
        If the mandatorry input files are not found.
    KeyError
        If the mandatory key are not found.
    UserWarning
        If missing data are found.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, root, patterns, channels, split="train", targets=None,
                 target_mapping=None, transforms=None, mask=None,
                 withdraw_subjects=None):

        # Sanity
        if not isinstance(patterns, (list, tuple)):
            patterns = [patterns]
        if not isinstance(channels, (list, tuple)):
            channels = [channels]
        if targets is not None and not isinstance(targets, (list, tuple)):
            targets = [targets]
        participant_file = os.path.join(root, "participants.tsv")
        split_file = os.path.join(root, f"{split}.tsv")
        for path in (participant_file, split_file):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), path)

        # Parameters
        self.root = root
        self.patterns = patterns
        self.channels = channels
        self.n_modalities = len(self.channels)
        self.targets = targets
        self.target_mapping = target_mapping or {}
        self.split = split
        self.transforms = transforms
        self.mask = (np.load(mask) if mask is not None else None)

        # Load subjects
        self.info_df = pd.read_csv(participant_file, sep="\t")
        if "participant_id" not in self.info_df:
            raise KeyError(
                "A 'participant_id' is mandatory in the participants file.")
        self.info_df = self.info_df.astype({"participant_id": "str"})
        self.split_df = pd.read_csv(split_file, sep="\t")
        if "participant_id" not in self.split_df:
            raise KeyError(
                "A 'participant_id' is mandatory in the split file.")
        self.split_df = self.split_df[["participant_id"]]
        self.split_df = self.split_df.astype({"participant_id": "str"})
        if withdraw_subjects is not None:
            self.split_df = self.split_df[
                ~self.split_df["participant_id"].isin(withdraw_subjects)]
        self._df = pd.merge(self.split_df, self.info_df, on="participant_id")

        # Keep only useful information / sanitize
        if targets is not None:
            for key in targets:
                if key not in self._df:
                    raise KeyError(
                        f"A '{key}' column is mandatory in the participant "
                        "file.")
        self._df = self._df[["participant_id"] + (targets or [])]
        _missing_data = self._df[self._df.isnull().any(axis=1)]
        if len(_missing_data) > 0:
            warnings.warn(f"Missing data in {split}!", UserWarning,
                          stacklevel=2)
        self._df.replace(self.target_mapping, inplace=True)
        self._targets = (
            self._df[targets].values if targets is not None else None)

    def __repr__(self):
        return (f"{self.__class__.__name__}<split='{self.split}',"
                f"modalities={self.n_modalities},targets={self.targets}>")

    def __len__(self):
        return len(self._df)


class BaseNumpyDataset(BaseDataset):
    """ Neuroimaging dataset that uses numpy arrays and memory mapping.

    Notes
    -----
    A 'participants.tsv' file containing subject information (including the
    requested targets) is expected at the root.
    A '<split>.tsv' file containg the subject to include is expected at the
    root.

    Parameters
    ----------
    root: str
        the location where are stored the data.
    patterns: str or list of str
        the relative locations (no path names matching allowed in specified
        pattern) of the numpy array to be loaded.
    channels: str or list of str, default=None
        the name of the channels.
    split: str, default 'train'
        define the split to be considered.
    targets: str or list of str, default=None
        the dataset will also return these tabular data.
    target_mapping: dict, default None
        optionaly, define a dictionary specifying different replacement values
        for different existing values. See pandas DataFrame.replace
        documentation for more information.
    transforms: callable, default None
        a function that can be called to augment the input images.
    mask: str, default None
        optionnaly, mask the input data using this numpy array.
    withdraw_subjects: list of str, default None
        optionaly, provide a list of subjects to remove from the dataset.

    Raises
    ------
    FileNotFoundError
        If the mandatorry input files are not found.
    KeyError
        If the mandatory key are not found.
    UserWarning
        If missing data are found.
    """
    def __init__(self, root, patterns, channels, split="train", targets=None,
                 target_mapping=None, transforms=None, mask=None,
                 withdraw_subjects=None):
        super().__init__(
            root, patterns, channels, split=split, targets=targets,
            target_mapping=target_mapping, transforms=transforms, mask=mask,
            withdraw_subjects=withdraw_subjects)
        self._data = [np.load(os.path.join(root, name), mmap_mode="r")
                      for name in patterns]

    def get_data(self, idx):
        """ Proper data indexing.
        """
        subject = self._df.iloc[idx].participant_id
        data_idx = self.info_df.loc[
            self.info_df.participant_id == subject].index.item()
        if self._targets is None:
            return [arr[data_idx] for arr in self._data], None
        else:
            return ([arr[data_idx] for arr in self._data],
                    (self._targets[idx]
                     if self._targets is not None else None))

    @abc.abstractmethod
    def __getitem__(self, idx):
        """ Get an item of the dataset: this method must be implemented in
        derived class.
        """


class BaseImageDataset(BaseDataset):
    """ Scalable neuroimaging dataset that uses files.

    Notes
    -----
    A 'participants.tsv' file containing subject information (including the
    requested targets) is expected at the root.
    A '<split>.tsv' file containg the subject to include is expected at the
    root.
    The general idea is not to copy all your data in the root folder but rather
    use a single symlink per project (if you are working with aggregated
    data). To enforce reproducibility you can check if the content of
    each file is persistent using the `get_checksum` method.

    Parameters
    ----------
    root: str
        the location where are stored the data.
    patterns: str or list of str
        the relative locations of the images to be loaded.
    channels: str or list of str, default=None
        the name of the channels.
    subject_in_patterns: int or list of int
        the folder level where the subject identifiers can be retrieved.
    split: str, default 'train'
        define the split to be considered.
    targets: str or list of str, default=None
        the dataset will also return these tabular data.
    target_mapping: dict, default None
        optionaly, define a dictionary specifying different replacement values
        for different existing values. See pandas DataFrame.replace
        documentation for more information.
    transforms: callable, default None
        a function that can be called to augment the input images.
    mask: str, default None
        optionnaly, mask the input data using this numpy array.
    withdraw_subjects: list of str, default None
        optionaly, provide a list of subjects to remove from the dataset.

    Raises
    ------
    FileNotFoundError
        If the mandatorry input files are not found.
    KeyError
        If the mandatory key are not found.
    UserWarning
        If missing data are found.
    """
    def __init__(self, root, patterns, channels, subject_in_patterns,
                 split="train", targets=None, target_mapping=None,
                 transforms=None, mask=None, withdraw_subjects=None):
        super().__init__(
            root, patterns, channels, split=split, targets=targets,
            target_mapping=target_mapping, transforms=transforms, mask=mask,
            withdraw_subjects=withdraw_subjects)

        if not isinstance(subject_in_patterns, (list, tuple)):
            subject_in_patterns = [subject_in_patterns]
        assert len(patterns) == len(subject_in_patterns)

        self.subject_in_patterns = subject_in_patterns

        self._data = {}
        for idx, pattern in enumerate(patterns):
            _regex = os.path.join(root, pattern)
            _sidx = subject_in_patterns[idx]
            _files = {
                self.sanitize_subject(path.split(os.sep)[_sidx]): path
                for path in glob.glob(_regex)}
            self._data[f"{self.channels[idx]}"] = [
                _files.get(subject) for subject in self._df["participant_id"]]
        self._data = pd.DataFrame.from_dict(self._data)
        _missing_data = self._data[self._data.isnull().any(axis=1)]
        if len(_missing_data) > 0:
            warnings.warn(f"Missing file data in {split}!", UserWarning,
                          stacklevel=2)
        self._data = self._data.values

    def sanitize_subject(self, subject):
        return subject.replace("sub-", "").split("_")[0]

    def get_checksum(self, path):
        """ Hashing file.
        """
        with open(path) as of:
            checksum = hashlib.sha1(of.read()).hexdigest()
        return checksum

    def get_data(self, idx):
        """ Proper data indexing.
        """
        return self._data[idx], (self._targets[idx]
                                 if self._targets is not None else None)

    @abc.abstractmethod
    def __getitem__(self, idx):
        """ Get an item of the dataset: this method must be implemented in
        derived class.
        """
