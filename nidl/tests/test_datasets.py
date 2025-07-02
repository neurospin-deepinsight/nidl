##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import unittest
from unittest.mock import patch

import pandas as pd
import torch

from nidl.datasets.base import (
    BaseImageDataset,
    BaseNumpyDataset,
)
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


if __name__ == "__main__":
    unittest.main()
