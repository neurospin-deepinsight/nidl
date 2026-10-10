##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from __future__ import annotations

import csv
import fnmatch
import glob
import gzip
import itertools
import json
import os
import random
import re
import struct
import tarfile
import warnings
from collections.abc import Sequence
from operator import itemgetter
from typing import Any, Callable, Optional, Union

import nibabel
import numpy as np
from torch.utils.data import Dataset

# preprocessing -> (folder in the repository, available modalities)
PREPROCESSINGS = {
    "quasiraw": ("quasiraw-v2.1.1", ("t1", "t2", "flair")),
    "cat12vbm": ("cat12vbm-12.8.2", ("mwp1",)),
}
MODALITIES = tuple(
    dict.fromkeys(m for _, mods in PREPROCESSINGS.values() for m in mods)
)
INDEX_NAME = "index.json.gz"
_INDEX_VERSION = 1
_DATASET_RE = re.compile(r"ds\d+")
_MEMBER_RE = re.compile(
    r"(?P<key>.+)\.(?P<label>(?:meta|morphometry)(?=\.json$)"
    r"|(?:t1|t2|flair|mwp1)(?:_mask)?(?=\.nii(?:\.gz)?$))"
)


class OpenMind(Dataset):
    """OpenMind dataset [1]_, read from WebDataset shards.

    The OpenMind dataset is a large-scale 3D MRI dataset of the head and neck
    region featuring 114k MRI Images. This data was pooled from exactly 800
    datasets from the OpenNeuro platform and provides 23 different MRI
    modalities from over 30 different scanners, representing a highly variable
    pre-training dataset.

    The brain T1w, T2w and FLAIR images were preprocessed with `brainprep
    <https://github.com/brainprepdesk/brainprep>`_ and are available with
    two preprocessings (see `preprocessing`):

    - "quasiraw": 42941 T1w, 23935 T2w and 4614 FLAIR images, with minimal
      preprocessing (skull-stripping, N4 bias field correction, 1mm isotropic
      resampling and linear registration to MNI space), shape
      `(1, 182, 218, 182)`.
    - "cat12vbm": T1w images processed with CAT12 VBM (v12.8.2). The images
      are the modulated gray matter maps ("mwp1") non-linearly registered to
      MNI space at 1.5mm isotropic, shape `(1, 113, 137, 113)`. Each image
      comes with its morphometry: the gray matter volume of each region of
      16 atlases (see `return_morphometry`).

    The shards contain one sample per (subject, session, modality, run) tuple.
    Deciding which images go together is done here, at load time, with
    `group_by`, `require_all` and `instance`.

    .. warning::
        The quasi-raw images take ~460GB of disk. Keep `streaming=True` if
        you intend to use only a small portion of the dataset.

    Parameters
    ----------
    root : str
        Root directory of the dataset, mirroring the `Hugging Face repository
        <https://huggingface.co/datasets/neurospin/openmind>`_: one folder per
        preprocessing (e.g. ``<root>/quasiraw-v2.1.1/``) containing one folder
        of ``.tar`` shards per OpenNeuro dataset, and ``val_split.csv``.
        Missing files are fetched from the Hugging Face.

    preprocessing : {"quasiraw", "cat12vbm"}, default="quasiraw"
        Which preprocessed images to load.

    modality : None, str or sequence of str, default=None
        Modalities to keep, among "t1", "t2" and "flair" for the quasi-raw
        preprocessing and "mwp1" for CAT12 VBM. None keeps all of them. A
        string returns a single image per item; a sequence returns a dict
        {modality: image} per item, see `group_by`.

    group_by : None, {"session", "subject"} or callable, default=None
        How images are regrouped into items:

        - None: one item per image, whatever its modality.
        - "session": one item per (dataset, subject, session), holding one
          image per requested modality (e.g. T1w + T2w acquired in the same
          session).
        - "subject": same, across all the sessions of a subject. Subject
          labels are only unique within a dataset, so the dataset id (e.g.
          "ds000030", parsed from the shard name) is part of the group.
        - callable: takes the metadata dict of an image (keys "dataset",
          "subject", "session", "modality", "run" and, for the quasi-raw
          images, "entities") and returns a hashable group key.

        Without grouping, an item is always a single array, even if several
        modalities are requested. With grouping, an item is a dict
        {modality: image} unless `modality` is a single string.

    require_all : bool, default=True
        Only relevant with `group_by`. If True, groups that miss one of the
        requested modalities are dropped. If False, they are kept and the
        missing modalities are set to None (use a custom `collate_fn` with a
        `DataLoader`).

    instance : {"first", "random", "all"}, default="first"
        Only relevant with `group_by`. When a group holds several images of
        the same modality (several runs, acquisitions...):

        - "first": keep the first one in sorted key order (the image
          without run/acq/... entities comes first).
        - "random": draw one at each `__getitem__` call.
        - "all": one item per combination (cartesian product).

    subjects : sequence of str or (str, str), default=None
        Keep only these subjects, given either as "sub-01" (matches this
        label in every dataset) or as ("ds000030", "sub-10159"). Subject
        labels are only unique within a dataset. Combined with `split`
        (both filters apply).

    split : {None, "train", "val"}, default=None
        Which subject-level split to load. None loads everything. The split
        is read from `split_file`, which lists the subjects of the "val"
        split; every other subject belongs to "train". A subject, with all
        its sessions, modalities and instances, is entirely in one split, so
        there is no leakage between splits, including with `group_by`.

    split_file : str or None, default=None
        CSV file listing the "val" subjects, with (at least) the columns
        ``dataset_id`` and ``participant_id``. Other columns (e.g. session,
        run) are ignored: the split is decided per (dataset, subject).
        Default is ``<root>/val_split.csv``, downloaded from the Hugging Face
        if missing. Only used if `split` is not None.

    apply_mask : bool, default=False
        If True, multiply each image by its brain mask. Images without mask
        (e.g. CAT12 VBM) are returned unchanged.

    return_morphometry : bool, default=False
        Only available with ``preprocessing="cat12vbm"``. If True,
        `__getitem__` returns ``(data, morphometry)`` where `morphometry` is
        a dict {atlas: {region: gray matter volume}} (plus the "total_volumes"
        TIV, GM, WM and CSF volumes). For grouped dict items, `morphometry` is
        a dict {modality: morphometry}.

    streaming : bool, default=True
        If True, the images of the shards missing locally are downloaded on
        demand from the Hugging Face (when accessed via `__getitem__`), one
        image at a time with HTTP range requests: nothing is written to disk.
        If False, all the shards holding the selected images are downloaded
        at initialization (whole shards, ≤3GB each).

    max_workers : int, default=1
        Number of concurrent threads to download the shards with
        `streaming=False` (1 thread = 1 shard download). Setting it > 1 can
        raise Hugging Face 429 errors (too many requests).

    shards : str, default="*/*.tar"
        Glob pattern selecting the shards, relative to the preprocessing
        folder (e.g. "ds0000*/*.tar"). A warning is raised if the same image
        appears in several of them.

    transforms : callable or None, default=None
        A function/transform that takes in an image (or a dict of images) and
        returns a transformed version.

    drop_cache : bool, default=False
        If True, ask the kernel to drop the page cache of each image right
        after reading it from a local shard
        (``posix_fadvise(POSIX_FADV_DONTNEED)``). An epoch streams through the
        whole dataset once, so caching it brings nothing but memory pressure:
        on shared filesystems such as Lustre, a page cache filling the job
        memory limit makes its reclaim, and hence the dataloader workers, very
        slow.

    Examples
    --------
    One quasi-raw T1w image per item:

    >>> dataset = OpenMind(root="data/openmind", modality="t1")
    >>> t1 = dataset[0]
    >>> print(t1.shape)
    (1, 182, 218, 182)

    Paired T1w/T2w acquired in the same session:

    >>> dataset = OpenMind(
    ...     root="data/openmind",
    ...     modality=("t1", "t2"),
    ...     group_by="session",
    ... )
    >>> sample = dataset[0]
    >>> print(sample["t1"].shape, sample["t2"].shape)
    (1, 182, 218, 182) (1, 182, 218, 182)

    CAT12 VBM gray matter maps with their morphometry:

    >>> dataset = OpenMind(root="data/openmind", preprocessing="cat12vbm",
    ...                    return_morphometry=True)
    >>> mwp1, morphometry = dataset[0]
    >>> print(mwp1.shape, morphometry["total_volumes"]["TIV"])
    (1, 113, 137, 113) 1391.08099215686

    Subject-level train/validation splits (from ``<root>/val_split.csv``):

    >>> train = OpenMind(root="data/openmind", modality="t1", split="train")
    >>> val = OpenMind(root="data/openmind", modality="t1", split="val")

    One image per subject, drawn among all modalities:

    >>> dataset = OpenMind(
    ...     root="data/openmind",
    ...     group_by=lambda meta: meta["subject"],
    ...     require_all=False,
    ...     instance="random",
    ... )

    Notes
    -----
    The data are downloaded exclusively from the `OpenMind repository
    <https://huggingface.co/datasets/neurospin/openmind>`_ on the Hugging Face
    either on-the-fly (lazy download) or during initialization (immediate
    download) if they are not already there.

    Images are read by byte offsets, listed in an index file
    (``<root>/<preprocessing folder>/index.json.gz``) released with the
    shards. If this index is neither available locally nor on the Hugging
    Face, it is built from the local shards (see `build_index`). Random access
    thus works with any sampler and any number of `DataLoader` workers, and
    the ``webdataset`` package is not needed to read the data.

    References
    ----------
    .. [1] Wald et al., An OpenMind for 3D medical vision self-supervised
           learning, arXiv 2025
    """

    REPO_ID = "neurospin/openmind"
    REVISION = "ed0af62645c4e8230f82e28a74b0dfa7bd6bfacb"  # commit ID

    def __init__(
        self,
        root: str,
        preprocessing: str = "quasiraw",
        modality: Optional[Union[str, Sequence[str]]] = None,
        group_by: Optional[Union[str, Callable[[dict], Any]]] = None,
        require_all: bool = True,
        instance: str = "first",
        subjects: Optional[Sequence[str]] = None,
        split: Optional[str] = None,
        split_file: Optional[str] = None,
        apply_mask: bool = False,
        return_morphometry: bool = False,
        streaming: bool = True,
        max_workers: int = 1,
        shards: str = "*/*.tar",
        transforms: Optional[Callable] = None,
        drop_cache: bool = False,
    ):
        self.root = self._parse_root(root)
        self.preprocessing = self._parse_preprocessing(preprocessing)
        self.folder = PREPROCESSINGS[self.preprocessing][0]
        self.modalities = self._parse_modality(modality)
        self.group_by = self._parse_group_by(group_by)
        self.require_all = require_all
        if instance not in ("first", "random", "all"):
            raise ValueError(
                f"instance must be 'first', 'random' or 'all', got "
                f"{instance!r}"
            )
        self.instance = instance
        self.split = self._parse_split(split)
        self.apply_mask = apply_mask
        if return_morphometry and self.preprocessing != "cat12vbm":
            raise ValueError(
                "return_morphometry is only available with "
                "preprocessing='cat12vbm'"
            )
        self.return_morphometry = return_morphometry
        self.streaming = streaming
        self.transforms = transforms
        self.drop_cache = drop_cache and hasattr(os, "posix_fadvise")
        # A dict per item only makes sense when regrouping several modalities
        self._as_dict = group_by is not None and not isinstance(modality, str)

        self.shards, self.entries = self._load_index()
        in_shards = {
            i for i, s in enumerate(self.shards) if fnmatch.fnmatch(s, shards)
        }
        if not in_shards:
            raise ValueError(
                f"No shard matching {shards!r} in the {self.preprocessing} "
                f"index"
            )

        keep = set(subjects) if subjects is not None else None
        val_subjects = None
        if self.split is not None:
            val_subjects = self._load_val_subjects(
                split_file
                or self.get_or_download_file(
                    os.path.join(self.root, "val_split.csv")
                )
            )
        self._warn_duplicates(
            [e for e in self.entries if e["shard"] in in_shards]
        )
        self._selected = [
            i
            for i, e in enumerate(self.entries)
            if e["shard"] in in_shards
            and e["meta"]["modality"] in self.modalities
            and (keep is None or self._is_kept(e["meta"], keep))
            and (
                val_subjects is None
                or self._in_val(e["meta"], val_subjects)
                == (self.split == "val")
            )
        ]
        self.items = self._make_items()
        if not self.streaming:  # fetch all the needed shards if not there
            self.download_shards(max_workers=max_workers)
        # shard index -> file descriptor, or -1 if read from the Hugging Face
        self._handles: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        """Load one item.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve.

        Returns
        -------
        data : array or dict of array
            A float32 array of shape `(1, X, Y, Z)`, or a dict
            {modality: array} for grouped multimodal items.
        morphometry : dict, optional
            Only if `return_morphometry` is True.
        """
        chosen = {}
        for modality, candidates in self.items[idx].items():
            if not candidates:
                chosen[modality] = None
            elif self.instance == "random" and len(candidates) > 1:
                chosen[modality] = random.choice(candidates)
            else:
                chosen[modality] = candidates[0]

        images, morphometries = {}, {}
        for modality, entry_idx in chosen.items():
            if entry_idx is None:
                images[modality], morphometries[modality] = None, None
                continue
            entry = self.entries[entry_idx]
            images[modality] = self._load_image(entry)
            if self.return_morphometry:
                morphometries[modality] = self._load_morphometry(entry)

        if self._as_dict:
            data, morphometry = images, morphometries
        else:
            # Exactly one modality per item (either ungrouped or a single
            # requested modality).
            (data,) = images.values()
            morphometry = next(iter(morphometries.values()), None)
        if self.transforms is not None:
            data = self.transforms(data)
        return (data, morphometry) if self.return_morphometry else data

    def __str__(self) -> str:
        return (
            f"OpenMind(root={self.root!r}, "
            f"preprocessing={self.preprocessing!r}, "
            f"modalities={self.modalities}, group_by={self.group_by!r}, "
            f"instance={self.instance!r}, n_items={len(self)})"
        )

    def __getstate__(self):
        # File handles cannot be pickled (DataLoader workers, spawn).
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __del__(self):
        for fd in getattr(self, "_handles", {}).values():
            if fd >= 0:
                os.close(fd)

    def download_file(self, filename: str) -> str:
        """Download a single file from the OpenMind repository on the HF."""
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

    def download_shards(self, max_workers: int = 1):
        """Fetch the shards holding the selected images from Hugging Face if
        not present.

        Only the missing shards are downloaded.

        Parameters
        ----------
        max_workers: int, default=1
            Number of concurrent threads to download files (1 thread = 1 file
            download).
        """
        needed = sorted({self.entries[i]["shard"] for i in self._selected})
        missing = [
            f"{self.folder}/{self.shards[shard]}"
            for shard in needed
            if not os.path.isfile(self._shard_path(shard))
        ]
        if not missing:
            return
        try:
            from huggingface_hub import snapshot_download
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{len(missing)} shards cannot be downloaded from Hugging "
                "Face because the 'huggingface_hub' package is not installed. "
                "Please run 'pip install huggingface_hub' first."
            ) from e
        snapshot_download(
            repo_id=self.REPO_ID,
            revision=self.REVISION,
            repo_type="dataset",
            local_dir=self.root,
            allow_patterns=missing,
            max_workers=max_workers,
        )

    def get_or_download_file(self, path: str) -> str:
        if not os.path.isfile(path):
            # Download the file from HF
            self.download_file(os.path.relpath(path, self.root))
        return path

    @staticmethod
    def build_index(data_dir: str, shards: str = "*/*.tar") -> str:
        """Index the local shards of one preprocessing.

        Record the byte offsets of every image (and mask, morphometry) of the
        shards, so that they can be read by random access, and write them to
        ``<data_dir>/index.json.gz``. This file is released with the shards
        on the Hugging Face: it only has to be rebuilt when the shards change.

        Parameters
        ----------
        data_dir : str
            Folder of a preprocessing (e.g. ``<root>/quasiraw-v2.1.1``).

        shards : str, default="*/*.tar"
            Glob pattern (relative to `data_dir`) selecting the shards to
            index.

        Returns
        -------
        path : str
            Path to the written index.
        """
        paths = sorted(glob.glob(os.path.join(data_dir, shards)))
        if not paths:
            raise FileNotFoundError(
                f"No shard matching {shards!r} found in {data_dir}"
            )
        entries = []
        for shard_idx, path in enumerate(paths):
            entries.extend(OpenMind._index_shard(path, shard_idx))
        index = {
            "version": _INDEX_VERSION,
            "shards": [os.path.relpath(p, data_dir) for p in paths],
            "entries": entries,
        }
        index_path = os.path.join(data_dir, INDEX_NAME)
        tmp = f"{index_path}.{os.getpid()}.tmp"
        with gzip.open(tmp, "wt") as f:
            json.dump(index, f)
        os.replace(tmp, index_path)  # atomic: safe with several ranks
        return index_path

    def _load_index(self) -> tuple[list[str], list[dict]]:
        """Load the index of the shards, fetching it if missing."""
        data_dir = os.path.join(self.root, self.folder)
        path = os.path.join(data_dir, INDEX_NAME)
        if not os.path.isfile(path):
            try:
                self.download_file(f"{self.folder}/{INDEX_NAME}")
            except (OSError, ModuleNotFoundError) as e:
                # Not released (yet) or offline: index the local shards.
                if not glob.glob(os.path.join(data_dir, "*/*.tar")):
                    raise FileNotFoundError(
                        f"The {self.preprocessing} index could not be "
                        f"downloaded from the Hugging Face and no local "
                        f"shard was found in {data_dir}"
                    ) from e
                warnings.warn(
                    f"The {self.preprocessing} index could not be downloaded "
                    f"from the Hugging Face ({e.__class__.__name__}): "
                    f"building it from the local shards in {data_dir}",
                    stacklevel=3,
                )
                self.build_index(data_dir)
        with gzip.open(path, "rt") as f:
            index = json.load(f)
        if index.get("version") != _INDEX_VERSION:
            raise ValueError(
                f"{path} has version {index.get('version')}, expected "
                f"{_INDEX_VERSION}: delete it to fetch or build it again"
            )
        return index["shards"], index["entries"]

    def _make_items(self) -> list[dict[str, tuple[int, ...]]]:
        """Regroup entries into items: {modality: candidate entry indices}."""
        if self.group_by is None:
            return [
                {self.entries[i]["meta"]["modality"]: (i,)}
                for i in self._selected
            ]

        groups: dict[Any, dict[str, list[int]]] = {}
        for i in self._selected:
            meta = self.entries[i]["meta"]
            group = groups.setdefault(self.group_by(meta), {})
            group.setdefault(meta["modality"], []).append(i)

        items = []
        for group in groups.values():
            if self.require_all and any(
                m not in group for m in self.modalities
            ):
                continue
            # Missing modalities get an empty candidate list -> None
            group = {
                m: tuple(
                    sorted(
                        group.get(m, ()), key=lambda i: self.entries[i]["key"]
                    )
                )
                for m in self.modalities
            }
            if self.instance == "first":
                items.append({m: c[:1] for m, c in group.items()})
            elif self.instance == "random":
                items.append(group)
            else:  # "all"
                pools = [c or (None,) for c in group.values()]
                items.extend(
                    {
                        m: (() if c is None else (c,))
                        for m, c in zip(group, combo)
                    }
                    for combo in itertools.product(*pools)
                )
        return items

    def _shard_path(self, shard: int) -> str:
        return os.path.join(self.root, self.folder, self.shards[shard])

    def _read(self, shard: int, offset: int, size: int) -> bytes:
        fd = self._handles.get(shard)
        if fd is None:
            path = self._shard_path(shard)
            fd = os.open(path, os.O_RDONLY) if os.path.isfile(path) else -1
            self._handles[shard] = fd
        if fd < 0:
            return self._read_remote(shard, offset, size)
        # Unbuffered positional reads on raw file descriptors
        chunks, n_read = [], 0
        while n_read < size:
            chunk = os.pread(fd, size - n_read, offset + n_read)
            if not chunk:
                raise EOFError(
                    f"Unexpected end of {self._shard_path(shard)} at byte "
                    f"{offset + n_read}"
                )
            chunks.append(chunk)
            n_read += len(chunk)
        if self.drop_cache:
            os.posix_fadvise(fd, offset, size, os.POSIX_FADV_DONTNEED)
        return chunks[0] if len(chunks) == 1 else b"".join(chunks)

    def _read_remote(self, shard: int, offset: int, size: int) -> bytes:
        """Read a byte range of a shard on the HF (HTTP range request)."""
        filename = f"{self.folder}/{self.shards[shard]}"
        try:
            from huggingface_hub import hf_hub_url
            from huggingface_hub.utils import (
                build_hf_headers,
                hf_raise_for_status,
                http_backoff,
            )
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"File '{filename}' cannot be streamed from Hugging Face "
                "because the 'huggingface_hub' package is not installed. "
                "Please run 'pip install huggingface_hub' first."
            ) from e
        url = hf_hub_url(
            self.REPO_ID,
            filename,
            repo_type="dataset",
            revision=self.REVISION,
        )
        headers = build_hf_headers()
        headers["Range"] = f"bytes={offset}-{offset + size - 1}"
        response = http_backoff(
            "GET",
            url,
            headers=headers,
            retry_on_status_codes=(429, 500, 502, 503, 504),
        )
        hf_raise_for_status(response)
        if response.status_code != 206 or len(response.content) != size:
            raise OSError(
                f"Invalid range request on {url}: got {len(response.content)}"
                f" bytes (HTTP {response.status_code}), expected {size}"
            )
        return response.content

    def _load_image(self, entry: dict) -> np.ndarray:
        image = self._decode(self._read(entry["shard"], *entry["image"]))
        if self.apply_mask and entry["mask"] is not None:
            mask = self._decode(self._read(entry["shard"], *entry["mask"]))
            image = image * (mask > 0)
        return image

    def _load_morphometry(self, entry: dict) -> Optional[dict]:
        if entry["morphometry"] is None:
            return None
        return json.loads(self._read(entry["shard"], *entry["morphometry"]))

    @staticmethod
    def _decode(buf: bytes) -> np.ndarray:
        """Decode the bytes of a .nii or .nii.gz file to `(1, X, Y, Z)`."""
        # Some shards were written with a double gzip layer.
        while buf[:2] == b"\x1f\x8b":
            buf = gzip.decompress(buf)
        # sizeof_hdr is 348 for NIfTI-1 and 540 for NIfTI-2, in either
        # endianness.
        sizes = {struct.unpack(f"{e}i", buf[:4])[0] for e in "<>"}
        cls = nibabel.Nifti2Image if 540 in sizes else nibabel.Nifti1Image
        data = cls.from_bytes(buf).get_fdata(dtype=np.float32)
        return data[None]

    @staticmethod
    def _index_shard(path: str, shard_idx: int) -> list[dict]:
        """Record the byte offsets of every image of a shard.

        Two layouts are supported: one modality per sample (``modality`` in
        ``meta.json``, files ``<modality>.nii.gz`` and
        ``<modality>_mask.nii.gz``), or several modalities bundled in one
        sample (``t1.nii.gz``, ``t2.nii.gz``... next to a single
        ``meta.json``). Either way, one entry is created per image.
        """
        found = _DATASET_RE.search(os.path.basename(path))
        dataset = found.group(0) if found else None
        samples: dict[str, dict] = {}
        with tarfile.open(path, "r:") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                # Match known suffixes rather than splitting at the first dot
                # like webdataset: keys may contain dots (e.g. "ses-1.5T").
                found = _MEMBER_RE.match(os.path.basename(member.name))
                if found is None:
                    continue
                key, label = found.group("key"), found.group("label")
                sample = samples.setdefault(key, {})
                if label == "meta":
                    sample["meta"] = json.load(tar.extractfile(member))
                else:
                    sample[label] = (member.offset_data, member.size)

        entries = []
        for key, sample in samples.items():
            meta = sample.get("meta")
            if meta is None:
                continue
            modalities = (
                [meta["modality"]]
                if "modality" in meta
                else [m for m in MODALITIES if m in sample]
            )
            for modality in modalities:
                if modality not in sample:
                    continue  # mask without its image
                entries.append(
                    {
                        "key": key,
                        "shard": shard_idx,
                        "meta": {
                            "run": None,
                            "entities": None,
                            **meta,
                            "modality": modality,
                            "dataset": dataset,
                        },
                        "image": sample[modality],
                        "mask": sample.get(f"{modality}_mask"),
                        "morphometry": sample.get("morphometry"),
                    }
                )
        return entries

    @staticmethod
    def _warn_duplicates(entries: list[dict]) -> None:
        """Warn if the same image is indexed twice (e.g. a stale shard)."""
        seen: dict[tuple, int] = {}
        for e in entries:
            m = e["meta"]
            ident = (
                m["dataset"],
                m["subject"],
                m["session"],
                m["modality"],
                m["run"],
                tuple(sorted((m["entities"] or {}).items())),
            )
            seen[ident] = seen.get(ident, 0) + 1
        dups = [i for i, n in seen.items() if n > 1]
        if dups:
            warnings.warn(
                f"{len(dups)} images are present in several shards, e.g. "
                f"{dups[0]}. Restrict `shards` to avoid duplicates.",
                stacklevel=3,
            )

    @staticmethod
    def _load_val_subjects(path: str) -> set:
        """Read the (dataset, subject) pairs of the "val" split from a CSV."""
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Split file not found: {path}. Pass `split_file` to use "
                f"another location."
            )
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            columns = {"dataset_id", "participant_id"}
            missing = columns - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{path} must have the columns {sorted(columns)}, "
                    f"missing {sorted(missing)}"
                )
            return {(r["dataset_id"], r["participant_id"]) for r in reader}

    @staticmethod
    def _in_val(meta: dict, val_subjects: set) -> bool:
        return (meta["dataset"], meta["subject"]) in val_subjects

    @staticmethod
    def _is_kept(meta: dict, keep: set) -> bool:
        return (
            meta["subject"] in keep
            or (meta["dataset"], meta["subject"]) in keep
        )

    @staticmethod
    def _parse_root(path: str) -> str:
        # eventually parse "~" or $HOME
        path = os.path.abspath(os.path.expanduser(path))
        # check if exists, otherwise create it
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _parse_preprocessing(preprocessing: str) -> str:
        if preprocessing not in PREPROCESSINGS:
            raise ValueError(
                f"preprocessing must be in {sorted(PREPROCESSINGS)}, got "
                f"{preprocessing!r}"
            )
        return preprocessing

    def _parse_modality(self, modality) -> tuple[str, ...]:
        available = PREPROCESSINGS[self.preprocessing][1]
        if modality is None:
            return available
        modalities = (modality,) if isinstance(modality, str) else modality
        modalities = tuple(m.lower() for m in modalities)
        unknown = set(modalities) - set(available)
        if unknown or not modalities:
            raise ValueError(
                f"Unknown modality {sorted(unknown)} for the "
                f"{self.preprocessing} preprocessing, expected among "
                f"{available}"
            )
        return modalities

    @staticmethod
    def _parse_split(split) -> Optional[str]:
        if split not in (None, "train", "val"):
            raise ValueError(
                f"split must be None, 'train' or 'val', got {split!r}"
            )
        return split

    @staticmethod
    def _parse_group_by(group_by) -> Optional[Callable[[dict], Any]]:
        if group_by is None or callable(group_by):
            return group_by
        if group_by == "session":
            return itemgetter("dataset", "subject", "session")
        if group_by == "subject":
            return itemgetter("dataset", "subject")
        raise ValueError(
            f"group_by must be None, 'session', 'subject' or a callable, "
            f"got {group_by!r}"
        )
