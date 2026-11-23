##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

r"""Unit tests for `nidl.datasets.OpenMind`."""

import csv
import gzip
import io
import json
import operator
import os
import shutil
import tarfile
import tempfile
import unittest
import warnings
from unittest.mock import MagicMock, patch

import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader

from nidl.datasets import OpenMind

SHAPE = (4, 5, 6)
QUASIRAW = "quasiraw-v2.1.1"
CAT12 = "cat12vbm-12.8.2"


def _nifti_bytes(value, shape=SHAPE, double_gzip=False, gz=True):
    """Build the (optionally gzipped) bytes of a small constant-valued
    NIfTI-1 image."""
    img = nib.Nifti1Image(np.full(shape, value, dtype=np.float32), np.eye(4))
    raw = img.to_bytes()
    if not gz:
        return raw
    data = gzip.compress(raw)
    if double_gzip:
        data = gzip.compress(data)
    return data


def _add_member(tar, name, data):
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _write_sample(
    tar,
    key,
    modality,
    value,
    entities=None,
    mask_value=None,
    subject=None,
    session=None,
    gz=True,
    double_gzip=False,
):
    """Write one quasi-raw (image[, mask], meta.json) sample to `tar`."""
    ext = "nii.gz" if gz else "nii"
    _add_member(
        tar,
        f"{key}.{modality}.{ext}",
        _nifti_bytes(value, gz=gz, double_gzip=double_gzip),
    )
    if mask_value is not None:
        mask = np.zeros(SHAPE, dtype=np.float32)
        mask[:2] = mask_value
        img = nib.Nifti1Image(mask, np.eye(4))
        _add_member(
            tar,
            f"{key}.{modality}_mask.{ext}",
            gzip.compress(img.to_bytes()) if gz else img.to_bytes(),
        )
    meta = {
        "subject": subject,
        "session": session,
        "modality": modality,
        "run": (entities or {}).get("run"),
        "entities": entities or None,
    }
    _add_member(tar, f"{key}.meta.json", json.dumps(meta).encode())


def _write_cat12_sample(tar, dataset, subject, session, run, value):
    """Write one CAT12 VBM (mwp1, morphometry.json, qc.json, meta.json)
    sample to `tar`: no "modality" in meta.json, as in the real shards."""
    key = f"{subject}_{session}_mod-mwp1_run-{run}"
    meta = {
        "dataset": dataset,
        "subject": subject,
        "session": session,
        "run": run,
    }
    morphometry = {
        "total_volumes": {"TIV": 1000.0 + value},
        "aal3": {"Vgm_lPreCG": value, "Vgm_rPreCG": 2 * value},
        # keys grouped by tissue, regions in the same order for each tissue
        "neuromorphometrics": {
            "Vgm_Left Amygdala": value,
            "Vgm_3rd Ventricle": 0.0,
            "Vwm_Left Amygdala": 10 * value,
            "Vwm_3rd Ventricle": 0.0,
            "Vcsf_Left Amygdala": 100 * value,
            "Vcsf_3rd Ventricle": 1.0,
        },
    }
    qc = {"group_stats": {"IQR": 2.0, "qc": 1.0}}
    _add_member(tar, f"{key}.meta.json", json.dumps(meta).encode())
    _add_member(
        tar, f"{key}.morphometry.json", json.dumps(morphometry).encode()
    )
    _add_member(tar, f"{key}.mwp1.nii.gz", _nifti_bytes(value))
    _add_member(tar, f"{key}.qc.json", json.dumps(qc).encode())


def _meta(dataset, idx):
    """Metadata of the (first candidate) image(s) of item `idx`."""
    metas = {
        m: dataset.entries[c[0]]["meta"] if c else None
        for m, c in dataset.items[idx].items()
    }
    return metas if len(metas) > 1 else next(iter(metas.values()))


def _metas(dataset):
    return [_meta(dataset, i) for i in range(len(dataset))]


class TestOpenMind(unittest.TestCase):
    """Test `nidl.datasets.OpenMind` against a small synthetic WebDataset,
    laid out as the Hugging Face repository."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="nidl_openmind_")
        # One folder per preprocessing, holding one folder of shards per
        # dataset: <root>/<preprocessing>/<dataset>/<dataset>-NNNNNN.tar
        qr = os.path.join(cls.root, QUASIRAW)
        os.makedirs(os.path.join(qr, "ds000001"))
        os.makedirs(os.path.join(qr, "ds000002"))

        # --- ds000001: the bulk of the grouping/instance scenarios --------
        with tarfile.open(
            os.path.join(qr, "ds000001", "ds000001-quasiraw-000000.tar"), "w"
        ) as tar:
            # sub-01/ses-01: T1 (no entity) + T2, paired -> group_by="session"
            _write_sample(
                tar,
                "sub-01_ses-01_mod-t1",
                "t1",
                1.0,
                mask_value=1.0,
                subject="sub-01",
                session="ses-01",
            )
            _write_sample(
                tar,
                "sub-01_ses-01_mod-t2",
                "t2",
                2.0,
                subject="sub-01",
                session="ses-01",
            )
            # sub-01/ses-01: two extra T1 instances (run-01, run-02), to
            # exercise `instance="first"/"random"/"all"`. "first" must pick
            # the no-entity image above (sorted key order).
            _write_sample(
                tar,
                "sub-01_ses-01_mod-t1_run-02",
                "t1",
                12.0,
                entities={"run": "02"},
                subject="sub-01",
                session="ses-01",
            )
            _write_sample(
                tar,
                "sub-01_ses-01_mod-t1_run-01",
                "t1",
                11.0,
                entities={"run": "01"},
                subject="sub-01",
                session="ses-01",
            )
            # sub-02, no session, T1 only + a FLAIR -> require_all=False
            # scenario, and a bare-key sample (no dots, no session).
            _write_sample(
                tar, "sub-02_mod-t1", "t1", 3.0, subject="sub-02", session=None
            )
            _write_sample(
                tar,
                "sub-02_mod-flair",
                "flair",
                4.0,
                subject="sub-02",
                session=None,
            )
            # sub-03/ses-01: T2 only, and written uncompressed (.nii) to
            # cover that code path.
            _write_sample(
                tar,
                "sub-03_ses-01_mod-t2",
                "t2",
                5.0,
                subject="sub-03",
                session="ses-01",
                gz=False,
            )
            # sub-04/ses-01: T1 written with a double gzip layer
            _write_sample(
                tar,
                "sub-04_ses-01_mod-t1",
                "t1",
                6.0,
                subject="sub-04",
                session="ses-01",
                double_gzip=True,
            )

        # ds000002: same subject LABEL "sub-01" as ds000001, to check
        # that grouping/labels are scoped per dataset, not merged.
        with tarfile.open(
            os.path.join(qr, "ds000002", "ds000002-quasiraw-000000.tar"), "w"
        ) as tar:
            _write_sample(
                tar,
                "sub-01_ses-01_mod-t1",
                "t1",
                100.0,
                subject="sub-01",
                session="ses-01",
            )
            _write_sample(
                tar,
                "sub-01_ses-01_mod-t2",
                "t2",
                200.0,
                subject="sub-01",
                session="ses-01",
            )

        # A stray duplicate of ds000001's shard, to test the
        # duplicate-shard warning under the default `shards="*/*.tar"` glob.
        shutil.copy(
            os.path.join(qr, "ds000001", "ds000001-quasiraw-000000.tar"),
            os.path.join(qr, "ds000001", "ds000001-duplicate-000000.tar"),
        )

        # --- CAT12 VBM: ds000001/sub-01 with two runs, ds000002/sub-01 -----
        cat12 = os.path.join(cls.root, CAT12)
        for dataset, samples in (
            ("ds000001", [("sub-01", "17", 1.0), ("sub-01", "18", 2.0)]),
            ("ds000002", [("sub-01", "19", 3.0)]),
        ):
            os.makedirs(os.path.join(cat12, dataset))
            with tarfile.open(
                os.path.join(cat12, dataset, f"{dataset}-000000.tar"), "w"
            ) as tar:
                for subject, run, value in samples:
                    _write_cat12_sample(
                        tar, dataset, subject, "ses-DEFAULT", run, value
                    )

        for folder in (QUASIRAW, CAT12):
            OpenMind.build_index(os.path.join(cls.root, folder))

        # ds000001/sub-01 is "val"; ds000002/sub-01 (same label, other
        # dataset) is NOT, and must stay in "train".
        cls.split_file = os.path.join(cls.root, "val_split.csv")
        with open(cls.split_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset_id", "participant_id", "session"])
            w.writerow(["ds000001", "sub-01", "ses-01"])

        # Phenotypes of the CAT12 images (one row per image, "n/a" if
        # unknown), as in the released participants.tsv
        with open(os.path.join(cls.root, "participants.tsv"), "w") as f:
            f.write(
                "dataset_id\tparticipant_id\tsession\trun\tage\tsex\tbmi"
                "\trace\thandedness\thealth_status\n"
                "ds000001\tsub-01\tses-DEFAULT\t17\t26.0\tfemale\tn/a"
                "\tn/a\tright\thealthy\n"
                "ds000001\tsub-01\tses-DEFAULT\t18\t26.0\tfemale\tn/a"
                "\tn/a\tright\thealthy\n"
                "ds000002\tsub-01\tses-DEFAULT\t19\tn/a\tmale\t22.5"
                "\tasian\tn/a\tn/a\n"
            )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self):
        # No test may reach the Hugging Face.
        patcher = patch.object(
            OpenMind,
            "download_file",
            side_effect=FileNotFoundError("offline"),
        )
        self.download_file = patcher.start()
        self.addCleanup(patcher.stop)

    def _quasiraw_only(self, **kwargs):
        """Build an OpenMind restricted to the non-duplicated shards."""
        return OpenMind(self.root, shards="*/*-quasiraw-*.tar", **kwargs)

    def _cat12(self, **kwargs):
        return OpenMind(self.root, preprocessing="cat12vbm", **kwargs)

    def _roi(self, **kwargs):
        return OpenMind(self.root, preprocessing="cat12vbm_roi", **kwargs)

    def test_len_and_modality_filter(self):
        dataset = self._quasiraw_only()
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(self._quasiraw_only(modality="t1")), 6)
        self.assertEqual(len(self._quasiraw_only(modality="t2")), 3)
        self.assertEqual(len(self._quasiraw_only(modality="flair")), 1)
        self.assertEqual(len(self._quasiraw_only(modality=("t1", "flair"))), 7)

    def test_getitem_shape_dtype_and_value(self):
        dataset = self._quasiraw_only(modality="t2")
        sample = dataset[0]
        self.assertEqual(sample.shape, (1, *SHAPE))
        self.assertEqual(sample.dtype, np.float32)

    def test_double_gzip_layer_is_decoded(self):
        # sub-04's T1 was written with two gzip layers stacked.
        dataset = self._quasiraw_only(modality="t1")
        idx = next(
            i
            for i, m in enumerate(_metas(dataset))
            if m["subject"] == "sub-04"
        )
        self.assertTrue(np.allclose(dataset[idx], 6.0))

    def test_uncompressed_nii_is_decoded(self):
        dataset = self._quasiraw_only(modality="t2")
        idx = next(
            i
            for i, m in enumerate(_metas(dataset))
            if m["subject"] == "sub-03"
        )
        self.assertTrue(np.allclose(dataset[idx], 5.0))

    def test_group_by_session_multimodal(self):
        dataset = self._quasiraw_only(
            modality=("t1", "t2"), group_by="session"
        )
        # Only ds1/sub-01 and ds2/sub-01 have both T1 and T2 in the same
        # session; sub-02 (no T2), sub-03 (no T1) and sub-04 (no T2) don't.
        self.assertEqual(len(dataset), 2)
        sample = dataset[0]
        self.assertIsInstance(sample, dict)
        self.assertEqual(set(sample), {"t1", "t2"})
        for value in sample.values():
            self.assertEqual(value.shape, (1, *SHAPE))

    def test_group_by_session_require_all_false_keeps_none(self):
        dataset = self._quasiraw_only(
            modality=("t1", "t2"), group_by="session", require_all=False
        )
        self.assertGreater(len(dataset), 3)
        has_none = [
            dataset[i]
            for i in range(len(dataset))
            if any(v is None for v in dataset[i].values())
        ]
        self.assertTrue(has_none)
        sample = has_none[0]
        self.assertTrue(any(v is None for v in sample.values()))
        self.assertTrue(any(v is not None for v in sample.values()))

    def test_group_by_subject_does_not_mix_datasets(self):
        dataset = self._quasiraw_only(
            modality=("t1", "t2"), group_by="subject", require_all=False
        )
        for meta in _metas(dataset):
            datasets = {m["dataset"] for m in meta.values() if m is not None}
            self.assertEqual(len(datasets), 1)

    def test_custom_group_by_callable(self):
        dataset = self._quasiraw_only(
            modality="t1", group_by=lambda meta: meta["subject"]
        )
        # sub-01 (ds1) and sub-01 (ds2) collapse under a *label-only*
        # callable
        self.assertLess(len(dataset), len(self._quasiraw_only(modality="t1")))

    def test_bad_group_by_raises(self):
        with self.assertRaises(ValueError):
            self._quasiraw_only(group_by="not-a-valid-mode")

    def test_instance_first_is_deterministic_and_entity_free(self):
        dataset = self._quasiraw_only(
            modality="t1", group_by="session", instance="first"
        )
        idx, meta = next(
            (i, m)
            for i, m in enumerate(_metas(dataset))
            if m["subject"] == "sub-01" and m["dataset"] == "ds000001"
        )
        self.assertTrue(np.allclose(dataset[idx], 1.0))  # no-entity image
        self.assertIsNone(meta["entities"])

    def test_instance_all_yields_every_combination(self):
        first = self._quasiraw_only(
            modality="t1", group_by="session", instance="first"
        )
        every = self._quasiraw_only(
            modality="t1", group_by="session", instance="all"
        )
        self.assertGreater(len(every), len(first))

    def test_instance_random_draws_among_candidates(self):
        dataset = self._quasiraw_only(
            modality="t1", group_by="session", instance="random"
        )
        idx = next(
            i
            for i, m in enumerate(_metas(dataset))
            if m["subject"] == "sub-01" and m["dataset"] == "ds000001"
        )
        values = {round(float(dataset[idx].flat[0])) for _ in range(40)}
        self.assertGreater(len(values), 1)
        self.assertTrue(values <= {1, 11, 12})

    def test_bad_instance_raises(self):
        with self.assertRaises(ValueError):
            self._quasiraw_only(instance="not-a-valid-mode")

    def test_apply_mask(self):
        without = self._quasiraw_only(modality="t1")
        masked = self._quasiraw_only(modality="t1", apply_mask=True)
        idx = next(
            i
            for i, item in enumerate(without.items)
            if without.entries[item["t1"][0]]["mask"] is not None
        )
        a, b = without[idx], masked[idx]
        self.assertFalse(np.array_equal(a, b))
        self.assertTrue((b[:, 2:] == 0).all())
        self.assertTrue(np.array_equal(a[:, :2], b[:, :2]))

    def test_transforms_applied(self):
        dataset = self._quasiraw_only(
            modality="t1", transforms=lambda x: x[:, ::2]
        )
        self.assertEqual(dataset[0].shape, (1, 2, *SHAPE[1:]))

    def test_meta_fields(self):
        meta = _meta(self._quasiraw_only(modality="t2"), 0)
        self.assertTrue(
            {"subject", "session", "modality", "dataset", "entities", "run"}
            <= set(meta)
        )

    def test_subjects_filter_by_label(self):
        dataset = self._quasiraw_only(subjects=["sub-01"])
        # matches sub-01 in *both* ds000001 and ds000002
        datasets = {m["dataset"] for m in _metas(dataset)}
        self.assertEqual(datasets, {"ds000001", "ds000002"})

    def test_subjects_filter_by_dataset_subject_pair(self):
        dataset = self._quasiraw_only(subjects=[("ds000001", "sub-01")])
        datasets = {m["dataset"] for m in _metas(dataset)}
        self.assertEqual(datasets, {"ds000001"})

    def test_unknown_modality_raises(self):
        with self.assertRaises(ValueError):
            self._quasiraw_only(modality="t3")
        with self.assertRaises(ValueError):
            self._quasiraw_only(modality="mwp1")

    def test_unknown_preprocessing_raises(self):
        with self.assertRaises(ValueError):
            OpenMind(self.root, preprocessing="fmriprep")

    def test_no_shard_found_raises(self):
        with self.assertRaises(ValueError):
            OpenMind(self.root, shards="*-nonexistent-*.tar")

    def test_duplicate_shards_warn(self):
        # The default shards="*/*.tar" also matches ds000001's stray
        # duplicate shard.
        with self.assertWarns(UserWarning):
            OpenMind(self.root)

    def test_split_train_val_partition_and_are_disjoint(self):
        full = self._quasiraw_only()
        train = self._quasiraw_only(split="train", split_file=self.split_file)
        val = self._quasiraw_only(split="val", split_file=self.split_file)
        self.assertEqual(len(train) + len(val), len(full))
        self.assertGreater(len(val), 0)

    def test_split_file_defaults_to_root(self):
        explicit = self._quasiraw_only(split="val", split_file=self.split_file)
        default = self._quasiraw_only(split="val")
        self.assertEqual(explicit.items, default.items)
        self.download_file.assert_not_called()

    def test_split_file_is_downloaded_if_missing(self):
        root = tempfile.mkdtemp(prefix="nidl_openmind_")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        shutil.copytree(
            os.path.join(self.root, QUASIRAW), os.path.join(root, QUASIRAW)
        )

        def fake_download(filename):
            return shutil.copy(self.split_file, os.path.join(root, filename))

        self.download_file.side_effect = fake_download
        val = OpenMind(root, shards="*/*-quasiraw-*.tar", split="val")
        self.download_file.assert_called_once_with("val_split.csv")
        self.assertEqual(
            {(m["dataset"], m["subject"]) for m in _metas(val)},
            {("ds000001", "sub-01")},
        )

    def test_split_is_scoped_per_dataset(self):
        # ds000002/sub-01 shares its label with the "val" ds000001/sub-01
        # but must remain in "train": the split is (dataset, subject), not
        # the bare subject label.
        train = self._quasiraw_only(split="train", split_file=self.split_file)
        val = self._quasiraw_only(split="val", split_file=self.split_file)
        train_pairs = {(m["dataset"], m["subject"]) for m in _metas(train)}
        val_pairs = {(m["dataset"], m["subject"]) for m in _metas(val)}
        self.assertIn(("ds000002", "sub-01"), train_pairs)
        self.assertNotIn(("ds000002", "sub-01"), val_pairs)
        self.assertEqual(val_pairs, {("ds000001", "sub-01")})
        self.assertFalse(train_pairs & val_pairs)

    def test_split_combines_with_subjects_filter(self):
        dataset = self._quasiraw_only(
            split="val",
            split_file=self.split_file,
            subjects=[("ds000002", "sub-01")],
        )
        self.assertEqual(len(dataset), 0)

    def test_split_none_is_unchanged(self):
        self.assertEqual(
            len(self._quasiraw_only()),
            len(self._quasiraw_only(split=None)),
        )

    def test_bad_split_value_raises(self):
        with self.assertRaises(ValueError):
            self._quasiraw_only(split="test")

    def test_missing_split_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            self._quasiraw_only(split="val", split_file="/no/such/file.csv")

    def test_split_file_missing_columns_raises(self):
        bad_file = os.path.join(self.root, "bad_split.csv")
        with open(bad_file, "w", newline="") as f:
            csv.writer(f).writerows([["a", "b"], ["1", "2"]])
        with self.assertRaises(ValueError):
            self._quasiraw_only(split="val", split_file=bad_file)

    def test_dataloader_multimodal_batches(self):
        dataset = self._quasiraw_only(
            modality=("t1", "t2"), group_by="session", require_all=False
        )
        loader = DataLoader(
            dataset, batch_size=2, num_workers=2, collate_fn=lambda x: x
        )
        batch = next(iter(loader))
        self.assertEqual(len(batch), 2)
        for sample in batch:
            self.assertEqual(set(sample), {"t1", "t2"})

    # --- CAT12 VBM ---------------------------------------------------------

    def test_cat12_images(self):
        dataset = self._cat12()
        self.assertEqual(dataset.modalities, ("mwp1",))
        self.assertEqual(len(dataset), 3)
        self.assertEqual(
            sorted(float(dataset[i].flat[0]) for i in range(len(dataset))),
            [1.0, 2.0, 3.0],
        )
        meta = _meta(dataset, 0)
        self.assertEqual(meta["modality"], "mwp1")
        self.assertEqual(meta["session"], "ses-DEFAULT")

    def test_cat12_runs_are_not_duplicates(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._cat12()

    def test_cat12_group_by_session(self):
        first = self._cat12(group_by="session", instance="first")
        every = self._cat12(group_by="session", instance="all")
        self.assertEqual((len(first), len(every)), (2, 3))

    def test_cat12_roi_single_atlas(self):
        dataset = self._roi(atlas="aal3")
        self.assertEqual(dataset.atlases, ["aal3"])
        self.assertEqual(len(dataset), 3)
        for i in range(len(dataset)):
            rois = dataset[i]
            self.assertEqual((rois.shape, rois.dtype), ((1, 2), np.float32))
            self.assertEqual(rois[0, 1], 2 * rois[0, 0])

    def test_cat12_roi_several_atlases(self):
        dataset = self._roi(atlas=("aal3", "total_volumes"))
        values = sorted(
            (rois["aal3"][0, 0], rois["total_volumes"][0, 0])
            for rois in (dataset[i] for i in range(len(dataset)))
        )
        self.assertEqual(values, [(v, 1000.0 + v) for v in (1.0, 2.0, 3.0)])
        self.assertEqual(
            dataset.get_roi_labels(),
            {"aal3": ["lPreCG", "rPreCG"], "total_volumes": ["TIV"]},
        )

    def test_cat12_roi_grouped_dict(self):
        dataset = self._roi(
            atlas="aal3", modality=("mwp1",), group_by="session"
        )
        rois = dataset[0]
        self.assertEqual(set(rois), {"mwp1"})
        self.assertEqual(rois["mwp1"].shape, (1, 2))

    def test_cat12_roi_default_atlas_and_tissue(self):
        dataset = self._roi()
        self.assertEqual(dataset.atlases, ["neuromorphometrics"])
        self.assertEqual(dataset.tissues, ["gm"])
        self.assertEqual(dataset[0].shape, (1, 2))

    def test_cat12_roi_tissues(self):
        dataset = self._roi(tissue=("csf", "gm", "wm"))
        for i in range(len(dataset)):
            rois = dataset[i]
            self.assertEqual(rois.shape, (3, 2))
            value = rois[1, 0]
            np.testing.assert_array_equal(
                rois, [[100 * value, 1.0], [value, 0.0], [10 * value, 0.0]]
            )
        self.assertEqual(
            dataset.get_roi_labels(),
            {"neuromorphometrics": ["Left Amygdala", "3rd Ventricle"]},
        )

    def test_cat12_roi_tissue_with_total_volumes(self):
        # total_volumes ignores `tissue`
        rois = self._roi(
            atlas=("neuromorphometrics", "total_volumes"), tissue="wm"
        )[0]
        self.assertEqual(rois["neuromorphometrics"].shape, (1, 2))
        self.assertEqual(rois["total_volumes"].shape, (1, 1))

    def test_bad_tissue_raises(self):
        for tissue in ("grey", (), ("gm", "gm")):
            with self.assertRaises(ValueError):
                self._roi(tissue=tissue)
        with self.assertRaises(ValueError):
            self._roi(atlas="aal3", tissue="wm")  # aal3 has GM only
        with self.assertRaises(ValueError):
            self._cat12(tissue="gm")  # only with cat12vbm_roi

    def test_bad_atlas_raises(self):
        for atlas in ("unknown", (), ("aal3", "aal3")):
            with self.assertRaises(ValueError):
                self._roi(atlas=atlas)
        with self.assertRaises(ValueError):
            self._cat12(atlas="aal3")  # only with cat12vbm_roi
        with self.assertRaises(ValueError):
            self._cat12().get_roi_labels()

    def test_images_table(self):
        dataset = self._cat12()
        self.assertIsNone(dataset._images)  # lazy: nothing read at init
        images = dataset.images
        self.assertEqual(len(images), len(dataset))
        self.assertEqual(
            sorted(images.index), sorted(i for it in dataset.items
                                         for c in it.values() for i in c)
        )  # fmt: skip
        row = images.loc[dataset.items[0]["mwp1"][0]]
        self.assertEqual(row["modality"], "mwp1")
        ds2 = images[images["dataset"] == "ds000002"].iloc[0]
        self.assertTrue(np.isnan(ds2["age"]))
        self.assertEqual((ds2["sex"], ds2["bmi"]), ("male", 22.5))

    def test_images_table_without_phenotypes(self):
        # Images missing from participants.tsv get NaN phenotypes
        images = self._quasiraw_only().images
        self.assertEqual(len(images), 10)
        self.assertTrue(images["age"].isna().all())

    def test_target(self):
        dataset = self._cat12(target="age", target_transforms=operator.neg)
        targets = {}
        for i in range(len(dataset)):
            image, age = dataset[i]
            targets[float(image.flat[0])] = age
        self.assertEqual((targets[1.0], targets[2.0]), (-26.0, -26.0))
        self.assertTrue(np.isnan(targets[3.0]))

    def test_target_list_with_rois(self):
        dataset = self._roi(atlas="aal3", target=["sex", "handedness"])
        rois, target = dataset[0]
        self.assertEqual(rois.shape, (1, 2))
        self.assertEqual(set(target), {"sex", "handedness"})

    def test_target_grouped_dict(self):
        dataset = self._cat12(
            modality=("mwp1",), group_by="subject", target="sex"
        )
        sexes = sorted(dataset[i][1] for i in range(len(dataset)))
        self.assertEqual(sexes, ["female", "male"])

    def test_unknown_target_raises(self):
        # Only the phenotypes of participants.tsv are valid targets
        for target in ("weight", "dataset", ["age", "modality"], []):
            with self.assertRaises(ValueError):
                self._cat12(target=target)

    def test_cat12_split(self):
        val = self._cat12(split="val")
        train = self._cat12(split="train")
        self.assertEqual((len(val), len(train)), (2, 1))

    # --- Index and Hugging Face fetching -----------------------------------

    def _new_root(self, folders=(QUASIRAW,), index=False):
        """Copy the shards (and the index) of `folders` to a new root."""
        root = tempfile.mkdtemp(prefix="nidl_openmind_")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        for folder in folders:
            ignore = None if index else shutil.ignore_patterns("index.json.gz")
            shutil.copytree(
                os.path.join(self.root, folder),
                os.path.join(root, folder),
                ignore=ignore,
            )
        return root

    def test_index_is_downloaded_if_missing(self):
        root = self._new_root()

        def fake_download(filename):
            path = os.path.join(root, filename)
            shutil.copy(os.path.join(self.root, filename), path)
            return path

        self.download_file.side_effect = fake_download
        dataset = OpenMind(root, shards="*/*-quasiraw-*.tar")
        self.download_file.assert_called_once_with(f"{QUASIRAW}/index.json.gz")
        self.assertEqual(len(dataset), 10)

    def test_index_is_built_from_local_shards_if_not_on_hf(self):
        root = self._new_root()
        with self.assertWarnsRegex(UserWarning, "local shards"):
            dataset = OpenMind(root, shards="*/*-quasiraw-*.tar")
        self.assertTrue(
            os.path.isfile(os.path.join(root, QUASIRAW, "index.json.gz"))
        )
        self.assertEqual(len(dataset), 10)
        # Reused afterwards
        self.download_file.reset_mock()
        OpenMind(root, shards="*/*-quasiraw-*.tar")
        self.download_file.assert_not_called()

    def test_no_index_nor_shards_raises(self):
        root = tempfile.mkdtemp(prefix="nidl_openmind_")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        with self.assertRaises(FileNotFoundError):
            OpenMind(root)

    def test_outdated_index_raises(self):
        root = self._new_root(index=True)
        path = os.path.join(root, QUASIRAW, "index.json.gz")
        with gzip.open(path, "rt") as f:
            index = json.load(f)
        index["version"] = -1
        with gzip.open(path, "wt") as f:
            json.dump(index, f)
        with self.assertRaises(ValueError):
            OpenMind(root)

    def _fake_range_get(self, folder):
        """Serve HTTP range requests from the shards of `self.root`."""

        def get(method, url, headers, **kwargs):
            filename = url.split(f"/resolve/{OpenMind.REVISION}/")[1]
            start, end = map(int, headers["Range"][6:].split("-"))
            with open(os.path.join(self.root, filename), "rb") as f:
                f.seek(start)
                content = f.read(end - start + 1)
            return MagicMock(status_code=206, content=content)

        return get

    def test_streaming_reads_missing_shards_from_hf(self):
        root = self._new_root(folders=(CAT12,), index=True)
        os.remove(os.path.join(root, CAT12, "ds000002", "ds000002-000000.tar"))
        with patch(
            "huggingface_hub.utils.http_backoff",
            side_effect=self._fake_range_get(CAT12),
        ) as get:
            for kwargs in (
                {"preprocessing": "cat12vbm"},
                {"preprocessing": "cat12vbm_roi", "atlas": "aal3"},
            ):
                local = OpenMind(self.root, **kwargs)
                streamed = OpenMind(root, **kwargs)
                for i in range(len(local)):
                    self.assertTrue(np.array_equal(local[i], streamed[i]))
        # Only ds000002 (1 image + 1 morphometry) is read remotely
        self.assertEqual(get.call_count, 2)
        for call in get.call_args_list:
            self.assertIn(f"{CAT12}/ds000002/", call.args[1])

    def test_no_streaming_downloads_missing_shards(self):
        root = self._new_root(index=True)
        with patch("huggingface_hub.snapshot_download") as download:
            OpenMind(
                root,
                shards="*/*-quasiraw-*.tar",
                subjects=["sub-01"],
                streaming=False,
            )
            download.assert_not_called()  # all shards are local

            os.remove(
                os.path.join(
                    root, QUASIRAW, "ds000002", "ds000002-quasiraw-000000.tar"
                )
            )
            OpenMind(
                root,
                shards="*/*-quasiraw-*.tar",
                subjects=[("ds000001", "sub-01")],
                streaming=False,
            )
            download.assert_not_called()  # ds000002 is not needed

            OpenMind(
                root,
                shards="*/*-quasiraw-*.tar",
                subjects=["sub-01"],
                streaming=False,
                max_workers=3,
            )
            download.assert_called_once()
            kwargs = download.call_args.kwargs
            self.assertEqual(kwargs["repo_id"], OpenMind.REPO_ID)
            self.assertEqual(kwargs["revision"], OpenMind.REVISION)
            self.assertEqual(kwargs["local_dir"], root)
            self.assertEqual(kwargs["max_workers"], 3)
            self.assertEqual(
                kwargs["allow_patterns"],
                [f"{QUASIRAW}/ds000002/ds000002-quasiraw-000000.tar"],
            )


if __name__ == "__main__":
    unittest.main()
