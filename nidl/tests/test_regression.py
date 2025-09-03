##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import unittest

import numpy as np

from nidl.metrics import pearson_r, regression_report

class TestRegressionMetrics(unittest.TestCase):
    """ Test regression metrics.
    """
    def test_pearson_r(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 4, 5]
        score = pearson_r(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0)

        y_true = 2 * np.array([1, 2, 3, 4, 5])
        y_pred = [1, 2, 3, 4, 5]
        score = pearson_r(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0)

        y_true = [2, 3, 4, 5]
        y_pred = [1, 2, 3, 4, 5]
        with self.assertRaises(ValueError):
            pearson_r(y_true, y_pred)
        
        y_true = [2, 2, 2, 2]
        y_pred = [1, 2, 3, 4]
        self.assertTrue(np.allclose(pearson_r(y_true, y_pred), np.nan, equal_nan=True))

        y_true = [[2, 1], [2, 2], [2, 3], [2, 4]]
        y_pred = [[1, 1], [2, 2], [3, 3], [4, 4]]
        self.assertTrue(np.allclose(pearson_r(y_true, y_pred), np.nan, equal_nan=True))
        self.assertTrue(np.allclose(pearson_r(y_true, y_pred, multioutput="raw_values"), 
                                    np.array([np.nan, 1.0]), equal_nan=True))

        y_true = [2, 2, 2, 2]
        y_pred = [1, 2, 3, 4]
        self.assertEqual(pearson_r(y_true, y_pred, force_finite=True), 0.0)

        y_true = [2, 2, 2, 2]
        y_pred = [2, 2, 2, 2]
        self.assertEqual(pearson_r(y_true, y_pred, force_finite=True), 1.0)

        y_true = np.ones((10, 2))
        y_pred = np.ones((10, 2))
        self.assertTrue(np.allclose(pearson_r(y_true, y_pred, force_finite=True), 1.0))

        y_true = np.random.rand(10, 2)
        y_pred = np.random.rand(10, 2)
        self.assertEqual(pearson_r(y_true, y_pred), pearson_r(y_pred, y_true))
        self.assertTrue(np.allclose(
            pearson_r(y_true, y_pred, multioutput='raw_values'),
            pearson_r(y_pred, y_true, multioutput='raw_values')
        ))

    def test_regression_report(self):
        y_true = np.random.rand(10, 1).repeat(3, axis=1)
        y_pred = np.random.rand(10, 1).repeat(3, axis=1)
        report = regression_report(y_true, y_pred, 
                                   sample_weight=np.random.rand(10),
                                   output_dict=True)
        reg_metrics = ['MAE', 'MedAE', 'RMSE', 'MSE', 'R2', 'PCC', 'EVar']
        for i in range(3):
            for metric in reg_metrics:
                self.assertAlmostEqual(report[f'regressor {i}'][metric], report[f'average'][metric])

        report = regression_report(y_true, y_pred, output_dict=False)
        self.assertIsInstance(report, str)

        report = regression_report(y_true, y_pred, 
                                   target_names=[str(i) for i in range(3)],
                                   output_dict=True)
        self.assertIsInstance(report, dict)
        self.assertTrue(len(report) == 4)

        y_true = np.random.rand(10, 1)
        y_pred = np.random.rand(10, 1)
        report = regression_report(y_true, y_pred, output_dict=True)
        self.assertTrue(set(report.keys()) == set(reg_metrics)) # metrics are reported directly
