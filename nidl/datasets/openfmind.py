##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from __future__ import annotations

import gzip
import os
from collections.abc import Callable, Sequence
from functools import reduce
from typing import Any, Optional, Union

import nibabel
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

PREPROCESSINGS = ("preproc", "quasiraw")
KINDS = ("timeseries", "connectivity")
TASK_TYPES = ("rest", "task")
# Atlas -> number of regions (atlases.tsv), the same for every run
ATLASES = {
    "aal": 166,  # AAL3 labels (166 non-empty regions)
    "aicha": 384,
    "basc064": 64,
    "basc122": 122,
    "basc197": 197,
    "basc325": 325,
    "basc444": 444,
    "destrieux": 150,
    "difumo64": 64,
    "difumo128": 128,
    "difumo256": 256,
    "difumo512": 512,
    "difumo1024": 1024,
    "glasser": 360,
    "harvardoxfordcort": 48,
    "harvardoxfordsub": 21,
    "msdl": 39,
    "pauli": 16,
    "schaefer100": 100,
    "schaefer200": 200,
    "schaefer300": 300,
    "schaefer400": 400,
    "schaefer500": 500,
    "schaefer600": 600,
    "schaefer800": 800,
    "schaefer1000": 1000,
    "schaefer10017n": 100,
    "schaefer20017n": 200,
    "schaefer40017n": 400,
    "schaefer100017n": 1000,
    "schaefer400tians2": 432,
    "schaefer400tians3": 450,
    "smith10": 10,
    "smith20": 20,
    "smith70": 70,
    "tians1": 16,
    "tians2": 32,
    "tians3": 50,
    "yan100_yeo7": 100,
    "yan100_yeo17": 100,
    "yan100_kong17": 100,
    "yan200_yeo7": 200,
    "yan200_yeo17": 200,
    "yan200_kong17": 200,
    "yan400_yeo7": 400,
    "yan400_yeo17": 400,
    "yan400_kong17": 400,
    "yan800_yeo7": 800,
    "yan800_yeo17": 800,
    "yan800_kong17": 800,
    "yeo7": 7,
    "yeo17": 17,
}
# BIDS entities identifying a run (columns of the index and of qc.tsv)
ENTITIES = (
    "dataset", "sub", "ses", "task", "acq", "ce", "dir", "rec", "run",
    "echo", "part",
)  # fmt: skip
QC_COLUMNS = ("mean_fd", "max_fd", "frac_fd_above_threshold", "n_volumes")


class OpenFMind(Dataset):
    """Open-fMIND dataset [1]_, read from tar shards without `webdataset`.

    Open-fMIND is a large-scale fMRI corpus for pretraining brain foundation
    models: 45,731 runs (6,044 hours of scanning) of 11,379 subjects from 322
    OpenNeuro datasets, preprocessed with fMRIPrep 25.2.3 and parcellated
    with 52 atlases (7 to 1,024 regions). Each run comes in two versions (see
    `preprocessing`) and two kinds (see `kind`).

    One item is one run (rest or task), loaded for one or several atlases.

    .. warning::
        The whole dataset takes 4.35TB of disk, and one (preprocessing, kind,
        atlas) triplet takes from a few to ~100GB. Keep `streaming=True` if
        you intend to use only a small portion of the dataset.

    Parameters
    ----------
    root : str
        Root directory of the dataset, mirroring the `Hugging Face repository
        <https://huggingface.co/datasets/cbarbano/open-fmind>`_ (e.g.
        ``<root>/index/preproc_timeseries.parquet``,
        ``<root>/preproc/timeseries/atlas-schaefer400/rest/shard-0000.tar``).
        Missing files are fetched from the Hugging Face.

    preprocessing : {"preproc", "quasiraw"}, default="preproc"
        Which version of the runs to load:

        - "preproc": denoised timeseries (26 confounds regression, scrubbing
          of the frames with a framewise displacement > 0.5mm, detrending,
          0.01-0.1Hz band-pass filtering, per-region z-scoring). Runs with
          less than 50 volumes left after scrubbing are not available.
        - "quasiraw": raw parcellated fMRIPrep outputs.

    kind : {"timeseries", "connectivity"}, default="timeseries"
        What to load for each run and atlas:

        - "timeseries": the mean BOLD signal of each region, a float32 array
          of shape `(n_rois, n_timepoints)`. The number of time points varies
          across runs (see Examples to batch them).
        - "connectivity": the Pearson correlation between regions, a float32
          array of shape `(n_rois, n_rois)`.

    atlas : str or sequence of str, default="schaefer400"
        Atlases to load, with their number of regions:

        - "aal": 166 (AAL3, without its 4 labels empty by design)
        - "aicha": 384
        - "basc064", "basc122", "basc197", "basc325", "basc444": 64 to 444
        - "destrieux": 150
        - "difumo64", "difumo128", "difumo256", "difumo512", "difumo1024":
          64 to 1024 (probabilistic)
        - "glasser": 360
        - "harvardoxfordcort": 48, "harvardoxfordsub": 21
        - "msdl": 39 (probabilistic)
        - "pauli": 16
        - "schaefer<N>" (7 networks) with N in 100, 200, 300, 400, 500, 600,
          800 and 1000, "schaefer<N>17n" (17 networks) with N in 100, 200,
          400 and 1000: N regions
        - "schaefer400tians2": 432, "schaefer400tians3": 450 (Schaefer 400
          + Tian subcortex)
        - "smith10", "smith20", "smith70": 10 to 70 (ICA maps)
        - "tians1": 16, "tians2": 32, "tians3": 50
        - "yan<N>_<networks>" with N in 100, 200, 400 and 800 and networks
          in "yeo7", "yeo17" and "kong17": N regions
        - "yeo7": 7, "yeo17": 17

        The number of regions is the same for every run: regions outside the
        field of view of a run are kept, with a zero signal. A string returns
        a single array per item; a sequence returns a dict {atlas: array} per
        item, holding the same run seen through several parcellations. See
        `get_atlases` for the references.

    task_type : None, {"rest", "task"}, default=None
        Keep only resting-state ("rest") or task ("task") runs. None keeps
        both. Restricting it also restricts the shards to download.

    subjects : sequence of str or (str, str), default=None
        Keep only these subjects, given either as "sub-01" (matches this
        label in every dataset) or as ("ds000030", "sub-10159"). Subject
        labels are only unique within a dataset. Use it to build
        subject-level splits.

    query : str or None, default=None
        Expression passed to :meth:`pandas.DataFrame.query` to select runs
        on their metadata (see `runs`): the BIDS entities (``dataset``,
        ``sub``, ``ses``, ``task``, ``run``...), ``task_type`` and the motion
        estimates of ``qc.tsv`` (``mean_fd``, ``max_fd``,
        ``frac_fd_above_threshold``, ``n_volumes``), e.g.
        ``"mean_fd < 0.3 and n_volumes >= 200"``.

    target : None, str or list of str, default=None
        Metadata column(s) of `runs` returned with each item (e.g.
        "task_type", "dataset", "mean_fd"). If a string, the value is
        returned; if a list, a dict {column: value}. If None (the default,
        for self-supervised learning), only the data are returned.

    streaming : bool, default=True
        If True, the runs of the shards missing locally are downloaded on
        demand from the Hugging Face (when accessed via `__getitem__`), one
        run at a time with HTTP range requests: nothing is written to disk.
        If False, all the shards holding the selected runs are downloaded
        at initialization (whole shards, ≤3GB each).

    max_workers : int, default=1
        Number of concurrent threads to download the shards with
        `streaming=False` (1 thread = 1 shard download). Setting it > 1 can
        raise Hugging Face 429 errors (too many requests).

    transforms : callable or None, default=None
        A function/transform that takes in an array (or a dict of arrays)
        and returns a transformed version.

    target_transforms : callable or None, default=None
        A function/transform applied to the target(s).

    Attributes
    ----------
    runs : pandas.DataFrame
        Metadata of the selected runs, one row per item (in the same order):
        ``task_type``, the BIDS entities and the motion estimates of
        ``qc.tsv`` (NaN when missing).

    Examples
    --------
    Denoised Schaefer-400 resting-state timeseries:

    >>> dataset = OpenFMind(root="data/openfmind", atlas="schaefer400",
    ...                     task_type="rest")
    >>> ts = dataset[0]
    >>> print(ts.shape)
    (400, 150)

    Two parcellations of the same run, cropped to a fixed random window so
    that they can be batched with the default `collate_fn` of a `DataLoader`:

    >>> def crop(views, length=100):
    ...     n = min(v.shape[-1] for v in views.values())
    ...     start = np.random.randint(0, n - length + 1)
    ...     return {a: v[:, start:start + length] for a, v in views.items()}
    >>> dataset = OpenFMind(
    ...     root="data/openfmind",
    ...     atlas=("schaefer400", "difumo256"),
    ...     query="n_volumes >= 100",
    ...     transforms=crop,
    ... )
    >>> views = dataset[0]
    >>> print(views["schaefer400"].shape, views["difumo256"].shape)
    (400, 100) (256, 100)

    Connectivity matrices of low-motion runs, with their dataset of origin:

    >>> dataset = OpenFMind(
    ...     root="data/openfmind",
    ...     kind="connectivity",
    ...     atlas="difumo64",
    ...     query="mean_fd < 0.2",
    ...     target="dataset",
    ... )
    >>> conn, site = dataset[0]
    >>> print(conn.shape, site)
    (64, 64) ds000030

    Notes
    -----
    The data are downloaded exclusively from the `Open-fMIND repository
    <https://huggingface.co/datasets/cbarbano/open-fmind>`_ on the Hugging
    Face either on-the-fly (lazy download) or during initialization
    (immediate download) if they are not already there. Only the runs are
    read: the JSON sidecars (provenance, ROI labels...) are skipped.

    Runs are read by byte offsets, listed in the parquet index released with
    the shards (``<root>/index/<preprocessing>_<kind>.parquet``, read with
    `pyarrow`). Random access thus works with any sampler and any number of
    `DataLoader` workers.

    Each source dataset keeps its own license, listed by `get_licenses`:
    please respect them and cite the datasets you use.

    References
    ----------
    .. [1] Barbano, C. A., Marraffini, G., Wassermann, D. Open-fMIND: A
           Large-Scale Open-Access fMRI Corpus for Pretraining Brain
           Foundation Models. arXiv:2611.21215, 2026.
    """

    REPO_ID = "cbarbano/open-fmind"
    REVISION = "6511aad6165b1f12520e6facad49be808ad8fb1f"  # commit ID

    def __init__(
        self,
        root: str,
        preprocessing: str = "preproc",
        kind: str = "timeseries",
        atlas: Union[str, Sequence[str]] = "schaefer400",
        task_type: Optional[str] = None,
        subjects: Optional[Sequence[Union[str, tuple[str, str]]]] = None,
        query: Optional[str] = None,
        target: Union[str, list[str], None] = None,
        streaming: bool = True,
        max_workers: int = 1,
        transforms: Optional[Callable] = None,
        target_transforms: Optional[Callable] = None,
    ):
        self.root = self._parse_root(root)
        self.preprocessing = self._parse_choice(
            "preprocessing", preprocessing, PREPROCESSINGS
        )
        self.kind = self._parse_choice("kind", kind, KINDS)
        self.atlases = self._parse_atlas(atlas)
        self.task_type = self._parse_choice(
            "task_type", task_type, (None, *TASK_TYPES)
        )
        self.target = target
        self.streaming = streaming
        self.transforms = transforms
        self.target_transforms = target_transforms
        # A dict per item only makes sense when loading several atlases
        self._as_dict = not isinstance(atlas, str)

        runs, self.shards, locations = self._load_index()
        keep = self._select(runs, subjects, query)
        self.runs = runs[keep].reset_index(drop=True)
        # (n_atlases, n_runs) arrays: cheap to pickle to DataLoader workers
        self._shard, self._offset, self._size = (a[:, keep] for a in locations)
        if self.target is not None:
            missing = (
                {self.target}
                if isinstance(self.target, str)
                else set(self.target)
            )
            missing -= set(self.runs.columns)
            if missing:
                raise ValueError(
                    f"Unknown target {sorted(missing)}, expected among "
                    f"{list(self.runs.columns)}"
                )
        if not self.streaming:  # fetch all the needed shards if not there
            self.download_shards(max_workers=max_workers)
        # shard index -> file descriptor, or -1 if read from the Hugging Face
        self._handles: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self.runs)

    def __getitem__(self, idx: int):
        """Load one run.

        Parameters
        ----------
        idx : int
            Index of the run to retrieve.

        Returns
        -------
        data : array or dict of array
            A float32 array of shape `(n_rois, n_timepoints)` (timeseries)
            or `(n_rois, n_rois)` (connectivity), or a dict {atlas: array}
            if several atlases are requested.
        target : Any, optional
            Only if `target` is not None.
        """
        views = {
            atlas: self._decode(
                self._read(
                    int(self._shard[i, idx]),
                    int(self._offset[i, idx]),
                    int(self._size[i, idx]),
                )
            )
            for i, atlas in enumerate(self.atlases)
        }
        data = views if self._as_dict else views[self.atlases[0]]
        if self.transforms is not None:
            data = self.transforms(data)
        if self.target is None:
            return data
        row = self.runs.iloc[idx]
        target = (
            row[self.target]
            if isinstance(self.target, str)
            else row[list(self.target)].to_dict()
        )
        if self.target_transforms is not None:
            target = self.target_transforms(target)
        return data, target

    def __str__(self) -> str:
        return (
            f"OpenFMind(root={self.root!r}, "
            f"preprocessing={self.preprocessing!r}, kind={self.kind!r}, "
            f"atlases={self.atlases}, task_type={self.task_type!r}, "
            f"n_runs={len(self)})"
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
        """Download a single file from the Open-fMIND repository on the HF."""
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
        """Fetch the shards holding the selected runs from Hugging Face if
        not present.

        Only the missing shards are downloaded.

        Parameters
        ----------
        max_workers: int, default=1
            Number of concurrent threads to download files (1 thread = 1 file
            download).
        """
        missing = [
            self.shards[shard]
            for shard in np.unique(self._shard)
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

    def get_qc(self) -> pd.DataFrame:
        """Motion estimates of every run (``qc.tsv``), with the BIDS
        entities named as in the index ("" for an absent entity)."""
        path = self.get_or_download_file(os.path.join(self.root, "qc.tsv"))
        qc = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        qc = qc.rename(
            columns={
                "dataset_id": "dataset",
                "subject_id": "sub",
                "session_id": "ses",
                "run_id": "run",
            }
        )
        qc[list(ENTITIES)] = qc[list(ENTITIES)].replace("none", "")
        for col in QC_COLUMNS:
            qc[col] = pd.to_numeric(qc[col])
        return qc

    def get_atlases(self) -> pd.DataFrame:
        """Name, number of regions and reference of the 52 atlases."""
        path = os.path.join(self.root, "atlases.tsv")
        return pd.read_csv(self.get_or_download_file(path), sep="\t")

    def get_licenses(self) -> pd.DataFrame:
        """License, authors and DOI of each source OpenNeuro dataset."""
        path = os.path.join(self.root, "LICENSES.tsv")
        return pd.read_csv(self.get_or_download_file(path), sep="\t")

    def _load_index(
        self,
    ) -> tuple[pd.DataFrame, list[str], tuple[np.ndarray, ...]]:
        """Read the runs of the requested atlases from the parquet index.

        Returns
        -------
        runs : pandas.DataFrame
            One row per run available in all the requested atlases.
        shards : list of str
            Paths of the shards, relative to `root`.
        locations : tuple of 3 arrays of shape (n_atlases, n_runs)
            Shard index, byte offset and size of each run in each atlas.
        """
        path = self.get_or_download_file(
            os.path.join(
                self.root, "index", f"{self.preprocessing}_{self.kind}.parquet"
            )
        )
        filters = [("atlas", "in", self.atlases), ("ext", "==", "nii.gz")]
        if self.task_type is not None:
            filters.append(("task_type", "==", self.task_type))
        try:
            # Predicate pushdown: only the rows of the requested atlases are
            # read out of the ~4.7M rows of the index
            index = pd.read_parquet(
                path,
                columns=["atlas", "task_type", *ENTITIES, "shard", "offset",
                         "size"],
                filters=filters,
            )  # fmt: skip
        except ImportError as e:
            raise ImportError(
                "The Open-fMIND index cannot be read because the 'pyarrow' "
                "package is not installed. Please run 'pip install pyarrow' "
                "first."
            ) from e
        missing = set(self.atlases) - set(index["atlas"].unique())
        if missing:  # the atlases themselves are checked by `_parse_atlas`
            raise ValueError(
                f"No run of the atlas {sorted(missing)} in {path} with "
                f"task_type={self.task_type!r}"
            )

        # Join the atlases on the run (BIDS entities) and keep the runs
        # available for all of them
        per_atlas = [
            index[index["atlas"] == a].set_index(list(ENTITIES))
            for a in self.atlases
        ]
        common = reduce(
            lambda x, y: x.intersection(y), (d.index for d in per_atlas)
        ).sort_values()
        per_atlas = [d.loc[common] for d in per_atlas]
        codes, shards = pd.factorize(
            pd.concat([d["shard"] for d in per_atlas])
        )
        locations = (
            codes.reshape(len(self.atlases), -1).astype(np.int32),
            np.stack([d["offset"].to_numpy(np.int64) for d in per_atlas]),
            np.stack([d["size"].to_numpy(np.int64) for d in per_atlas]),
        )

        runs = per_atlas[0][["task_type"]].reset_index()
        runs = runs.merge(
            self.get_qc()[[*ENTITIES, *QC_COLUMNS]], on=list(ENTITIES),
            how="left",
        )  # fmt: skip
        return runs, list(shards), locations

    def _select(
        self,
        runs: pd.DataFrame,
        subjects: Optional[Sequence],
        query: Optional[str],
    ) -> np.ndarray:
        """Boolean mask of the runs passing all the filters."""
        keep = np.ones(len(runs), dtype=bool)
        if subjects is not None:
            pairs = pd.MultiIndex.from_frame(runs[["dataset", "sub"]])
            labels = {_strip(s) for s in subjects if isinstance(s, str)}
            tuples = {
                (d, _strip(s)) for d, s in (
                    s for s in subjects if not isinstance(s, str)
                )
            }  # fmt: skip
            keep &= runs["sub"].isin(labels).to_numpy() | pairs.isin(
                list(tuples)
            )
        if query is not None:
            keep &= runs.index.isin(runs.query(query).index)
        return keep

    def _shard_path(self, shard: int) -> str:
        return os.path.join(self.root, self.shards[shard])

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
        # Drop the pages from the page cache: an epoch streams through the
        # whole dataset once, so they would only fill the job memory limit
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, offset, size, os.POSIX_FADV_DONTNEED)
        return chunks[0] if len(chunks) == 1 else b"".join(chunks)

    def _read_remote(self, shard: int, offset: int, size: int) -> bytes:
        """Read a byte range of a shard on the HF (HTTP range request)."""
        filename = self.shards[shard]
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

    @staticmethod
    def _decode(buf: bytes) -> np.ndarray:
        """Decode a .nii(.gz) run to `(n_rois, n_timepoints)` (timeseries,
        stored as `(n_rois, 1, 1, n_timepoints)`) or `(n_rois, n_rois)`
        (connectivity, stored as `(n_rois, n_rois, 1, 1)`)."""
        if buf[:2] == b"\x1f\x8b":
            buf = gzip.decompress(buf)
        image = nibabel.Nifti1Image.from_bytes(buf)
        data = np.asarray(image.dataobj, dtype=np.float32)
        return data.reshape(data.shape[0], -1)

    @staticmethod
    def _parse_root(path: str) -> str:
        # eventually parse "~" or $HOME
        path = os.path.abspath(os.path.expanduser(path))
        # check if exists, otherwise create it
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _parse_choice(name: str, value: Any, choices: tuple) -> Any:
        if value not in choices:
            raise ValueError(f"{name} must be in {choices}, got {value!r}")
        return value

    @staticmethod
    def _parse_atlas(atlas: Union[str, Sequence[str]]) -> list[str]:
        atlases = [atlas] if isinstance(atlas, str) else list(atlas)
        atlases = [a.lower().removeprefix("atlas-") for a in atlases]
        if (
            not atlases
            or not set(atlases) <= set(ATLASES)
            or len(set(atlases)) != len(atlases)
        ):
            raise ValueError(
                f"atlas must be one or several distinct atlases among "
                f"{list(ATLASES)}, got {atlas!r}"
            )
        return atlases


def _strip(subject: str) -> str:
    """Subject label as stored in the index ("sub-01" -> "01")."""
    return subject.removeprefix("sub-")
