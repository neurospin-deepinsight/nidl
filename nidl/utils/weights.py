##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Various utilities to download model weights.
"""
from __future__ import annotations

import urllib.parse as urlparse
import urllib.request as urlrequest
import warnings
from pathlib import Path
from typing import Any, Optional, Union

import torch

from ..estimators.base import BaseEstimator


class Weights:
    """A class to handle (retrieve and apply) model weights or lightning
    checkpoints.

    Parameters
    ----------
    name: str
        the location of the model weights specified in the form
        `hf-hub:path/architecture_name@revision` if available in Hugging Face
        hub or `ns-hub:path/architecture_name` if available in the NeuroSpin
        hub or a path if avaiable in your local machine.
    data_dir: pathlib.Path or str
        path where data should be downloaded.
    filepath: str
        the path of the file in the repo. If path has '.ckpt' extension, it
        assumes it is a pytorch_lightning checkpoint.
    """

    HF_URL = "https://huggingface.co"
    NS_URL = "http://nsap.intra.cea.fr/neurospin-hub/"

    def __init__(self, name: str, data_dir: Union[str, Path], filepath: str):
        self.name = name
        self.data_dir = Path(data_dir)
        self.filepath = filepath
        self.dtype = name.split(":")[0] if ":" in name else "local"
        # Whether the file is a lightning checkpoint file
        self.is_lightning_ckpt = filepath.endswith('.ckpt')
        if self.dtype == "hf-hub":
            hf_id, hf_revision = self.hub_split(name)
            self.weight_file = self.hf_download(
                data_dir, hf_id, filepath, hf_revision
            )
        elif self.dtype == "ns-hub":
            ns_id, _ = self.hub_split(name)
            self.weight_file = self.ns_download(data_dir, ns_id, filepath)
        else:
            self.weight_file = Path(self.name) / self.filepath
        assert self.weight_file.is_file(), (
            f"Invalid weights '{self.weight_file}'"
        )

    def load_checkpoint(self,
                        model: BaseEstimator,
                        #device: _DEVICE = 'cpu',
                        **kwargs: Any,
                        ):
        """Load the checkpoint.

        Parameters
        ----------
        model: LightningModule
            an pytorch_lightning's module class.
        device: torch.device or str
            the device on which to load the model and to use for inference.
            Default to cpu. Only single device is supported for now.
        **kwargs: Any extra keyword args needed to init the model. Can also be
            used to override saved hyperparameter values, in particular to
            override trainer parameters such as `accelerator` or `devices`.
        """
        if not self.is_lightning_ckpt:
            warnings.warn((
                f"The provided file ({self.weight_file}) does not seem to be "
                "a Lightning's checkpoint file. To load weights directly, use "
                "`load_pretrained` instead."),
                stacklevel=2)
            return
        # If `devices` and `accelerator` not in kwargs, add default values.
        # This is necessary to override the device and accelerator saved in the
        # checkpoint and ensure that the trainer within the estimator
        # will be instantiated with values compatible with the user"s setup.
        params_init = dict(
            kwargs,
            **({"devices": 'auto'} if 'devices' not in kwargs else {}),
            **({"accelerator": 'auto'} if 'accelerator' not in kwargs else {}),
        )
        return model.load_from_checkpoint(
            checkpoint_path=self.weight_file,
            map_location='cpu',
            **params_init)


    def load_pretrained(self, model: torch.nn.Module):
        """Load the model weights.

        Parameters
        ----------
        model: torch.nn.Module
            an input model with a `load_pretrained` method decalred.
        """
        if self.weight_file is None:
            warnings.warn("Define weight file location first!", stacklevel=2)
            return
        if self.is_lightning_ckpt:
            warnings.warn((
                "For pytorch_lightning checkpoints, use the method "
                "`load_checkpoint` instead."),
                stacklevel=2)
            return
        model.load_state_dict(torch.load(self.weight_file, weights_only=True))

    @classmethod
    def hub_split(cls, hub_name: str) -> tuple[str, Optional[str]]:
        """Interpret the input hub name specified in the form
        `hf-hub:path/architecture_name@revision` or
        `ns-hub:path/architecture_name`.

        Parameters
        ----------
        hub_name: str
            name of the repository.

        Returns
        -------
        hub_id: str
            the id of the repository.
        hub_revision: str
            the revision of the repository.
        """
        split = hub_name.split("@")
        assert 0 < len(split) <= 2, (
            "Hub name should be of the form "
            "`hub:path/architecture_name@revision`"
        )
        hub_id = split[0].split(":")[1]
        hub_revision = split[-1] if len(split) > 1 else None
        return hub_id, hub_revision

    @classmethod
    def hf_download(
        cls,
        data_dir: Union[str, Path],
        hf_id: str,
        filepath: str,
        hf_revision: Optional[str] = None,
        force_download: bool = False,
    ) -> Path:
        """Download a given file if not already present.

        Downloads always resume when possible. If you want to force a new
        download, use `force_download=True`.

        Parameters
        ----------
        data_dir: pathlib.Path or str
            path where data should be downloaded.
        hf_id: str
            the id of the repository.
        filepath: str
            the path of the file in the repo.
        hf_revision: str, default=None
            the revision of the repository (a tag, or a commit hash).
        force_download: bool, default=False
            whether the file should be downloaded even if it already exists in
            the local cache.

        Returns
        -------
        weight_file: Path
            local path to the model weights.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"File '{filepath}' cannot be downloaded from Hugging Face "
                "because the 'huggingface_hub' package is not installed. "
                "Please run 'pip install huggingface_hub' first."
            ) from e
        return Path(
            hf_hub_download(
                repo_id=hf_id,
                filename=filepath,
                revision=hf_revision,
                repo_type="model",
                local_dir=str(data_dir),
                force_download=force_download,
            )
        )

    @classmethod
    def ns_download(
        cls,
        data_dir: Union[str, Path],
        ns_id: str,
        filepath: str,
        force_download: bool = False,
    ) -> Path:
        """Download a given file if not already present.

        Downloads always resume when possible. If you want to force a new
        download, use `force_download=True`.

        Parameters
        ----------
        data_dir: pathlib.Path or str
            path where data should be downloaded.
        ns_id: str
            the id of the repository.
        filepath: str
            the path of the file in the repo.
        force_download: bool, default=False
            whether the file should be downloaded even if it already exists in
            the local cache.

        Returns
        -------
        weight_file: Path
            local path to the model weights.
        """
        split_id = ns_id.split("/")
        weight_file = Path(data_dir) / split_id[0] / split_id[1] / filepath
        if not force_download and weight_file.is_file():
            return weight_file
        weight_file.parent.mkdir(parents=True, exist_ok=True)
        url = urlparse.urljoin(cls.NS_URL, ns_id, filepath)
        urlrequest.urlretrieve(url, str(weight_file))
        return weight_file

    def __repr__(self):
        return (
            f"{self.__class__.__name__}<name={self.name},"
            f"data_dir={self.data_dir},filepath={self.filepath}>"
        )
