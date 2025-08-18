##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import unittest
from unittest.mock import patch, MagicMock
import os
import pandas as pd
import numpy as np
import shutil
import sys
import torch
from PIL import Image

from nidl.datasets.base import (
    BaseImageDataset,
    BaseNumpyDataset,
)
from nidl.datasets import OpenBHB, ImageDataFrameDataset
from nidl.datasets.pandas_dataset import default_loader
from nidl.utils import print_multicolor


class TestDatasets(unittest.TestCase):
    """ Test datasets.
    """
    def setUp(self):
        """ Setup test.
        """
        self.n_images = 10
        self.n_channels = 2
        self.fake_data = torch.rand(self.n_images, 128)
        _data = {
            "participant_id": ["000", "001", "002"],
            "target1": [3, 4, 2]
        }
        self.fake_df = pd.DataFrame(data=_data)
        _data = {
            "participant_id": ["001", "002"],
        }
        self.fake_train = pd.DataFrame(data=_data)

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def config(self):
        return {
            "root": "/mocked",
            "patterns": ["mocked"] * self.n_channels,
            "channels": [f"channel_{idx}" for idx in range(self.n_channels)],
            "split": "train",
            "targets": ["target1"],
            "target_mapping": None,
            "transforms": None,
            "mask": None,
            "withdraw_subjects": None
        }

    @patch("nidl.datasets.base.pd.read_csv")
    @patch("nidl.datasets.base.os.path.isfile")
    @patch("nidl.datasets.base.np.load")
    def test_numpy_dataset(self, mock_load, mock_isfile, mock_read_csv):
        """ Test numpy dataset.
        """
        mock_load.return_value = self.fake_data
        mock_isfile.return_value = True
        mock_read_csv.side_effect = [self.fake_df, self.fake_train]
        params = self.config()
        dataset = BaseNumpyDataset(**params)
        item = dataset.get_data(0)
        self.assertTrue(len(item) == 2)
        self.assertTrue(len(item[0]) == self.n_channels)
        self.assertTrue(all([_item.shape == (128, ) for _item in item[0]]))
        self.assertTrue(torch.allclose(item[0][0], self.fake_data[1]))
        self.assertTrue(item[1] == self.fake_df.loc[1].target1)

    @patch("nidl.datasets.base.pd.read_csv")
    @patch("nidl.datasets.base.os.path.isfile")
    @patch("nidl.datasets.base.glob.glob")
    def test_image_dataset(self, mock_glob, mock_isfile, mock_read_csv):
        """ Test image dataset.
        """
        mock_glob.side_effect = [
            [f"/mocked/sub-00{idx}/mod0" for idx in range(len(self.fake_df))],
            [f"/mocked/sub-00{idx}/mod1" for idx in range(len(self.fake_df))]
        ]
        mock_isfile.return_value = True
        mock_read_csv.side_effect = [self.fake_df, self.fake_train]
        params = self.config()
        dataset = BaseImageDataset(
            subject_in_patterns=-2,
            **params
        )
        item = dataset.get_data(0)
        self.assertTrue(len(item) == 2)
        self.assertTrue(len(item[0]) == self.n_channels)
        self.assertTrue(item[0].tolist() == [f"/mocked/sub-001/mod{idx}"
                                             for idx in range(2)])
        self.assertTrue(item[1] == self.fake_df.loc[1].target1)


class TestOpenBHB(unittest.TestCase):

    def setUp(self):
        """ Setup test.
        """
        self.local_path = "/tmp/openBHB"
        # Remove everything to ensure correct unit tests
        if os.path.exists(self.local_path):
            shutil.rmtree(self.local_path)
        self.splits_length = [('train', 3227), ('val', 757), ('internal_val', 362), ('external_val', 395)]
        self.modality_tests = [
            ("vbm", "age", (1, 121, 145, 121), float),
            ("quasiraw", "sex", (1, 182, 218, 182), str),
            ("vbm_roi", "age", (1, 284), float),
            ("fs_desikan_roi", "sex", (7, 68), str),
            ("fs_destrieux_roi", None, (7, 148), type(None)),
            ("fs_xhemi", "site", (8, 163842), int)
        ]
        self.multimodality_tests = [
            (["vbm", "quasiraw"], ["age", "sex"]),
            (["fs_destrieux_roi", "fs_desikan_roi"], ["sex", "site"]),
        ]
    
    def test_modalities_and_targets(self):
        for (modality, target, expected_shape, target_type) in self.modality_tests:
            dataset = OpenBHB(
                root=self.local_path,
                modality=modality,
                target=target,
                split="val",
                streaming=True
            )
            sample = dataset[0]
            if target is None:
                data = sample
            else:
                data, y = sample
                self.assertIsInstance(y, target_type)

            if isinstance(data, np.ndarray):
                self.assertEqual(data.shape, expected_shape)
            elif isinstance(data, dict):
                for k, v in data.items():
                    self.assertIsInstance(v, np.ndarray)
            else:
                self.fail(f"Unknown data type: {type(data)}")

    def test_multimodal_and_multitarget(self):
        for modalities, targets in self.multimodality_tests:
            dataset = OpenBHB(
                root=self.local_path,
                modality=modalities,
                target=targets,
                split="val",
                streaming=True
            )
            sample, target = dataset[0]
            self.assertIsInstance(sample, dict)
            self.assertIsInstance(target, dict)
            self.assertEqual(set(sample.keys()), set(modalities))
            self.assertEqual(set(target.keys()), set(targets))
            for key, img in sample.items():
                self.assertIsInstance(img, np.ndarray)
            for key in targets:
                self.assertIn(key, target)
    
    @patch("huggingface_hub.snapshot_download")
    def test_download_dataset_split_mocked(self, mock_snapshot_download):
        # Create a dummy OpenBHB instance with streaming=False
        dataset = OpenBHB(root=self.local_path, 
                          split="train", 
                          modality="vbm", 
                          streaming=False, 
                          max_workers=3)

        # Assert snapshot_download was called once
        mock_snapshot_download.assert_called_once()

        # Check the arguments it was called with
        kwargs = mock_snapshot_download.call_args.kwargs
        assert kwargs["repo_id"] == dataset.REPO_ID
        assert kwargs["revision"] == dataset.REVISION
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["local_dir"] == self.local_path
        assert len(kwargs["ignore_patterns"]) == 0
        assert "train/derivatives/sub-*/ses-*/sub-*cat12vbm_desc-gm_T1w.npy" in kwargs["allow_patterns"]
        assert kwargs["max_workers"] == 3
    
    def test_download_file_import_error(self):
        dataset = OpenBHB(root=self.local_path)

        # Simulate huggingface_hub not being installed
        with patch.dict(sys.modules, {"huggingface_hub": None}):
            with self.assertRaises(ModuleNotFoundError):
                dataset.download_file("some_file.npy")
            with self.assertRaises(ModuleNotFoundError):
                dataset.download_dataset_split("train", ("vbm",), [], incremental=False)
    
    def test_transforms(self):
        transform_tests = [
            (lambda x: x * 2, lambda y: y + 1),
            (lambda x: np.clip(x, 0, 1), lambda y: float(y)),
        ]
        for (transform, target_transform) in transform_tests:
            dataset = OpenBHB(
                root=self.local_path,
                modality="vbm_roi",
                target="age",
                split="val",
                transforms=transform,
                target_transforms=target_transform,
                streaming=True
            )
            x, y = dataset[0]
            self.assertIsInstance(x, np.ndarray)
            self.assertIsInstance(y, float)
    
    def test_masks_cover_images(self):
        split = "train"
        dataset = OpenBHB(root=self.local_path, split=split,
                            modality="vbm", target=None, streaming=True)
        for idx, sample in enumerate(dataset):
            image = sample[0]
            mask = dataset.get_cat12_template().get_fdata()  # adjust key if needed
            self.assertEqual(image.shape, mask.shape, 
                             f"Shape mismatch for sample {idx} in split {split}")

            # Check mask covers image
            coverage = np.sum(image[mask > 1e-7] > 0)
            self.assertEqual(coverage, np.sum(image > 0), 
                               f"Mask does not cover image at sample {idx} in split {split}")

            # Limit to 2 samples per split for speed
            if idx >= 2:
                break
    
    def test_splits_length(self):
        for (split, length) in self.splits_length:
            ds = OpenBHB(self.local_path, split=split)
            self.assertTrue(len(ds) == length)

    def test_templates_and_atlas(self):
        dataset = OpenBHB(
            root=self.local_path,
            modality="vbm",
            target="age",
            split="val",
            streaming=True
        )
        cat_template = dataset.get_cat12_template()
        raw_template = dataset.get_quasiraw_template()
        atlas = dataset.get_neuromorphometrics_atlas()
        labels_destrieux = dataset.get_fs_labels("destrieux")
        labels_desikan = dataset.get_fs_labels("desikan")
        labels_destrieux_sym = dataset.get_fs_labels("destrieux", symmetric=True)
        labels_desikan_sym = dataset.get_fs_labels("desikan", symmetric=True)

        self.assertEqual(cat_template.shape, (121, 145, 121))
        self.assertEqual(raw_template.shape, (182, 218, 182))
        self.assertIn("data", atlas)
        self.assertIn("labels", atlas)
        atlas_data = atlas["data"].get_fdata().astype(int)
        used_labels = set(np.unique(atlas_data))
        defined_labels = set(np.arange(len(atlas["labels"]), dtype=int))
        missing_labels = used_labels - defined_labels
        self.assertEqual(len(missing_labels), 0)
        self.assertEqual(len(labels_destrieux), 148) 
        self.assertEqual(len(labels_desikan), 68)
        self.assertEqual(len(labels_destrieux_sym), 74) 
        self.assertEqual(len(set(labels_destrieux_sym)), 74)
        self.assertEqual(len(labels_desikan_sym), 34)
        self.assertEqual(len(set(labels_desikan_sym)), 34)
        # Feature Names
        self.assertEqual(len(dataset.get_fs_roi_feature_names()), 7)
        self.assertEqual(len(dataset.get_fs_xhemi_feature_names()), 8)
        self.assertEqual(len(dataset.get_vbm_roi_labels()), 284)

    def test_str_repr(self):
        for mod in [["vbm"], ["vbm", "quasiraw"]]:
            dataset = OpenBHB(
                root=self.local_path,
                modality=mod,
                target="age",
                split="train",
                streaming=True
            )
            self.assertIn("openBHB", str(dataset))
            self.assertIn("-".join(mod), str(dataset))

    def test_invalid_split_raises(self):
        for split in ["testing", "", 123]:
            with self.assertRaises(ValueError):
                OpenBHB(root=self.local_path, modality="quasiraw", 
                        target=None, split=split, streaming=True)

    def test_invalid_modality_raises(self):
        for modality in ["badmod", 123, None, []]:
            with self.assertRaises(ValueError):
                OpenBHB(root=self.local_path, modality=modality, 
                        target="sex", split="train", streaming=True)

    def test_invalid_target_raises(self):
        for target in ["invalid", 123, []]:
            with self.assertRaises(ValueError):
                OpenBHB(root=self.local_path, modality="vbm", 
                        target=target, split="train", streaming=True)


class TestDefaultLoader(unittest.TestCase):
    @patch("nibabel.load")
    def test_load_nii(self, mock_nib):
        mock_img = MagicMock()
        mock_img.get_fdata.return_value = "nii_data"
        mock_nib.return_value = mock_img

        result = default_loader("test.nii")
        self.assertEqual(result, "nii_data")

    @patch("numpy.load")
    def test_load_numpy(self, mock_npy):
        mock_npy.return_value = np.array([1, 2, 3])
        result = default_loader("array.npy")
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    @patch("PIL.Image.open")
    def test_load_pil_image(self, mock_open):
        mock_img = MagicMock()
        mock_img.convert.return_value = "rgb_img"
        mock_open.return_value = mock_img

        result = default_loader("image.jpg")
        self.assertEqual(result, "rgb_img")

    def test_invalid_extension(self):
        with self.assertRaises(ValueError):
            default_loader("file.txt")


class TestImageDataFrameDataset(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "image_path": ["img1.jpg", "img2.jpg"],
            "label": ["cat", "dog"],
            "age": [5, 7]
        })

    def test_len_and_getitem_single_label(self):
        image_loader = lambda path: "image_data"
        ds = ImageDataFrameDataset(self.df, image_col="image_path", label_cols="label", 
                                    image_loader=image_loader)
        self.assertEqual(len(ds), 2)
        img, label = ds[0]
        self.assertEqual(img, "image_data")
        self.assertEqual(label, "cat")

    def test_getitem_multi_labels(self):
        image_loader = lambda path: "image_data"
        ds = ImageDataFrameDataset(self.df, image_col="image_path", label_cols=["label", "age"], 
                                   image_loader=image_loader)
        img, label = ds[1]
        self.assertEqual(img, "image_data")
        self.assertEqual(label, ["dog", 7])

    def test_transform_applied(self):
        image_loader = lambda path: "raw_img"
        transform = lambda x: f"processed_{x}"
        ds = ImageDataFrameDataset(self.df, transform=transform, label_cols="label", 
                                   image_loader=image_loader)
        img, label = ds[0]
        self.assertEqual(img, "processed_raw_img")

    def test_target_transform_callable(self):
        image_loader = lambda path: "img"
        target_transform = lambda y: y.upper()
        ds = ImageDataFrameDataset(self.df, label_cols="label", target_transform=target_transform,
                                    image_loader=image_loader)
        _, label = ds[0]
        self.assertEqual(label, "CAT")

    def test_target_transform_dict(self):
        image_loader = lambda path: "img"
        ttf = {"label": str.upper, "age": lambda x: x * 2}
        ds = ImageDataFrameDataset(self.df, label_cols=["label", "age"], target_transform=ttf,
                         image_loader=image_loader)
        _, label = ds[1]
        self.assertEqual(label, ["DOG", 14])

    def test_series_as_input(self):
        series = pd.Series(["img1.jpg", "img2.jpg"])
        ds = ImageDataFrameDataset(series)
        self.assertEqual(len(ds), 2)

    def test_csv_as_input(self):
        tmp_csv = "/tmp/temp.csv"
        self.df.to_csv(tmp_csv, index=False)
        ds = ImageDataFrameDataset(tmp_csv, image_col="image_path")
        self.assertEqual(len(ds), 2)

    def test_invalid_df_type(self):
        with self.assertRaises(TypeError):
            ImageDataFrameDataset(123)

    def test_missing_image_col(self):
        with self.assertRaises(ValueError):
            ImageDataFrameDataset(pd.DataFrame({"wrong": [1, 2]}))

    def test_missing_label_col(self):
        with self.assertRaises(ValueError):
            ImageDataFrameDataset(self.df, label_cols="nonexistent")

    def test_missing_label_cols_list(self):
        with self.assertRaises(ValueError):
            ImageDataFrameDataset(self.df, label_cols=["label", "nope"])

    def test_return_img_only_if_no_label(self):
        image_loader = lambda path: path
        ds = ImageDataFrameDataset(self.df, label_cols=None, return_none_if_no_label=False,
                                   image_loader=image_loader)
        img = ds[0]
        self.assertEqual(img, "img1.jpg")

    def test_invalid_file_filtering(self):
        df = pd.DataFrame({"image_path": ["valid.jpg", "invalid.txt"], "label": ["a", "b"]})
        ds = ImageDataFrameDataset(df, label_cols="label")
        self.assertEqual(len(ds), 1)
        self.assertIn("valid.jpg", ds.imgs)

    def test_invalid_label_filtering(self):
        df = pd.DataFrame({"image_path": ["img.jpg", "img2.jpg"], "label": ["ok", None]})
        ds = ImageDataFrameDataset(df, label_cols="label", is_valid_label=lambda x: x is not None)
        self.assertEqual(len(ds), 1)

    def test_indexing(self):
        image_loader = lambda path: "img"
        ds = ImageDataFrameDataset(self.df, label_cols="label", image_loader=image_loader)
        self.assertEqual(ds[0], ("img", "cat"))
        self.assertEqual(ds[1], ("img", "dog"))


if __name__ == "__main__":
    unittest.main()
