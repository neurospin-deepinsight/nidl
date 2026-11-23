##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

r"""Unit tests for `nidl.datasets.OpenFMind`."""

import gzip
import io
import json
import os
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from nidl.datasets import OpenFMind
from nidl.datasets.openfmind import ENTITIES

# atlas -> number of regions
ATLASES = {"msdl": 3, "yan100_yeo7": 2}
# (dataset, sub, task, task_type, n_timepoints, mean_fd, atlases)
RUNS = [
    ("ds000001", "01", "rest", "rest", 6, 0.1, ("msdl", "yan100_yeo7")),
    ("ds000001", "01", "nback", "task", 8, 0.5, ("msdl", "yan100_yeo7")),
    ("ds000002", "01", "rest", "rest", 7, 0.2, ("msdl", "yan100_yeo7")),
    ("ds000002", "02", "rest", "rest", 5, 0.3, ("msdl",)),
]


def _value(run_idx, atlas):
    return 10 * run_idx + list(ATLASES).index(atlas)


def _add_member(tar, name, data):
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))
    # offset_data of the member just written
    return tar.offset - _padded(len(data)), len(data)


def _padded(size):
    return (size + 511) // 512 * 512


def _make_repo(root, kind="timeseries"):
    """Write a fake Open-fMIND repository (shards, index, qc.tsv)."""
    rows, qc = [], []
    for atlas, n_rois in ATLASES.items():
        for task_type in ("rest", "task"):
            shard = f"preproc/{kind}/atlas-{atlas}/{task_type}/shard-0000.tar"
            os.makedirs(os.path.join(root, os.path.dirname(shard)),
                        exist_ok=True)  # fmt: skip
            with tarfile.open(os.path.join(root, shard), "w") as tar:
                for i, (ds, sub, task, tt, n_tp, _, atlases) in enumerate(
                    RUNS
                ):
                    if tt != task_type or atlas not in atlases:
                        continue
                    shape = (
                        (n_rois, 1, 1, n_tp)
                        if kind == "timeseries"
                        else (n_rois, n_rois, 1, 1)
                    )
                    img = nib.Nifti1Image(
                        np.full(shape, _value(i, atlas), np.float32),
                        np.eye(4),
                    )
                    key = (f"{ds}/sub-{sub}/func/sub-{sub}_task-{task}"
                           f"_atlas-{atlas}_desc-preproc_{kind}")  # fmt: skip
                    for ext, data in (
                        ("json", json.dumps({"TaskType": tt}).encode()),
                        ("nii.gz", gzip.compress(img.to_bytes())),
                    ):
                        offset, size = _add_member(tar, f"{key}.{ext}", data)
                        entities = dict.fromkeys(ENTITIES, "")
                        entities.update(dataset=ds, sub=sub, task=task)
                        rows.append({
                            "tree": "preproc", "kind": kind, "atlas": atlas,
                            "task_type": tt, **entities, "key": key,
                            "ext": ext, "member": f"{key}.{ext}",
                            "shard": shard, "offset": offset, "size": size,
                        })  # fmt: skip
    for ds, sub, task, _, n_tp, fd, _ in RUNS:
        qc.append({
            "dataset_id": ds, "subject_id": sub, "session_id": "none",
            "task": task, "run_id": "none", "acq": "none", "ce": "none",
            "dir": "none", "rec": "none", "echo": "none", "part": "none",
            "mean_fd": fd, "max_fd": 2 * fd,
            "frac_fd_above_threshold": 0.0, "n_volumes": n_tp,
        })  # fmt: skip
    os.makedirs(os.path.join(root, "index"), exist_ok=True)
    pd.DataFrame(rows).to_parquet(
        os.path.join(root, "index", f"preproc_{kind}.parquet")
    )
    pd.DataFrame(qc).to_csv(os.path.join(root, "qc.tsv"), sep="\t",
                            index=False)  # fmt: skip


class TestOpenFMind(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        _make_repo(self.root)
        # Nothing must be fetched from the Hugging Face
        patcher = patch.object(
            OpenFMind, "download_file", side_effect=AssertionError("network")
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_single_atlas(self):
        ds = OpenFMind(self.root, atlas="msdl")
        self.assertEqual(len(ds), len(RUNS))
        for idx in range(len(ds)):
            row = ds.runs.iloc[idx]
            i = next(
                k for k, r in enumerate(RUNS)
                if (r[0], r[1], r[2]) == (row.dataset, row["sub"], row.task)
            )  # fmt: skip
            ts = ds[idx]
            self.assertEqual(ts.dtype, np.float32)
            self.assertEqual(ts.shape, (ATLASES["msdl"], RUNS[i][4]))
            np.testing.assert_array_equal(ts, _value(i, "msdl"))
            self.assertAlmostEqual(row.mean_fd, RUNS[i][5])

    def test_multi_atlas_intersection(self):
        ds = OpenFMind(self.root, atlas=("msdl", "atlas-yan100_yeo7"))
        self.assertEqual(len(ds), 3)  # ds000002/sub-02 lacks yan100_yeo7
        views = ds[0]
        self.assertEqual(set(views), {"msdl", "yan100_yeo7"})
        self.assertEqual(views["msdl"].shape[0], 3)
        self.assertEqual(views["yan100_yeo7"].shape[0], 2)
        self.assertEqual(views["msdl"].shape[1], views["yan100_yeo7"].shape[1])

    def test_connectivity(self):
        _make_repo(self.root, kind="connectivity")
        ds = OpenFMind(self.root, kind="connectivity", atlas="yan100_yeo7")
        self.assertEqual(ds[0].shape, (2, 2))

    def test_filters(self):
        ds = OpenFMind(self.root, atlas="msdl", task_type="task")
        self.assertEqual(list(ds.runs.task), ["nback"])
        ds = OpenFMind(self.root, atlas="msdl", subjects=["sub-02"])
        self.assertEqual(len(ds), 1)
        ds = OpenFMind(self.root, atlas="msdl", subjects=[("ds000002", "01")])
        self.assertEqual(len(ds), 1)
        ds = OpenFMind(
            self.root,
            atlas="msdl",
            query="mean_fd < 0.25 and task_type == 'rest'",
        )
        self.assertEqual(len(ds), 2)

    def test_target_and_transforms(self):
        ds = OpenFMind(
            self.root, atlas="msdl", target="dataset",
            transforms=lambda x: x[:, :5].copy(), target_transforms=str.upper,
        )  # fmt: skip
        ts, target = ds[0]
        self.assertEqual(ts.shape, (3, 5))
        self.assertEqual(target, ds.runs.dataset[0].upper())
        ds = OpenFMind(self.root, atlas="msdl", target=["task_type", "sub"])
        self.assertEqual(set(ds[0][1]), {"task_type", "sub"})
        with self.assertRaises(ValueError):
            OpenFMind(self.root, atlas="msdl", target="age")

    def test_dataloader_workers(self):
        ds = OpenFMind(self.root, atlas=("msdl", "yan100_yeo7"),
                       transforms=lambda v: {a: x[:, :5] for a, x in
                                             v.items()})  # fmt: skip
        ds[0]  # open a file handle before pickling to the workers
        batches = list(DataLoader(ds, batch_size=2, num_workers=2))
        self.assertEqual(batches[0]["msdl"].shape, (2, 3, 5))
        self.assertEqual(sum(len(b["msdl"]) for b in batches), len(ds))

    def test_streaming_reads_remote(self):
        ds = OpenFMind(self.root, atlas="msdl")
        expected = ds[0]
        shard_path = ds._shard_path(int(ds._shard[0, 0]))
        with open(shard_path, "rb") as f:
            content = f.read()
        os.remove(shard_path)
        ds = OpenFMind(self.root, atlas="msdl")
        with patch.object(
            OpenFMind, "_read_remote",
            side_effect=lambda s, o, n: content[o:o + n],
        ) as remote:  # fmt: skip
            np.testing.assert_array_equal(ds[0], expected)
        remote.assert_called_once()

    def test_download_shards(self):
        ds = OpenFMind(self.root, atlas="msdl", task_type="rest")
        os.remove(ds._shard_path(0))
        with patch("huggingface_hub.snapshot_download") as download:
            OpenFMind(self.root, atlas="msdl", task_type="rest",
                      streaming=False)  # fmt: skip
        self.assertEqual(
            download.call_args.kwargs["allow_patterns"],
            ["preproc/timeseries/atlas-msdl/rest/shard-0000.tar"],
        )

    def test_unknown_atlas_raises_before_any_download(self):
        empty = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, empty)
        # download_file is patched to fail: the atlas is checked first
        for atlas in ("unknown", "schaefer450", ("msdl", "atlas-msdl")):
            with self.assertRaises(ValueError):
                OpenFMind(empty, atlas=atlas)
        self.assertEqual(os.listdir(empty), [])

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            OpenFMind(self.root, atlas="unknown")
        with self.assertRaises(ValueError):
            OpenFMind(self.root, preprocessing="raw")
        with self.assertRaises(ValueError):
            OpenFMind(self.root, kind="volume")
        with self.assertRaises(ValueError):
            OpenFMind(self.root, task_type="resting")
        with self.assertRaises(ValueError):
            OpenFMind(self.root, atlas=())


if __name__ == "__main__":
    unittest.main()
