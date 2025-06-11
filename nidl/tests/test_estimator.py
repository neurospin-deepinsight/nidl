##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


import glob
import os
import tempfile
import toml
import unittest
from nidl.estimators import (
    BaseEstimator, ClassifierMixin, ClusterMixin, RegressorMixin,
    TransformerMixin)
from nidl.utils import print_multicolor


class TestExperiment(unittest.TestCase):
    """ Test experiement.
    """
    def setUp(self):
        """ Setup test.
        """
        self.dirname = os.path.abspath(os.path.dirname(__file__))
        self.configs = glob.glob(os.path.join(
            self.dirname, "configs", "*.toml"))

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_mixin(self):
        """ Test Mixin types.
        """
        mro = BaseEstimator.__mro__
        print(f"[{print_multicolor(repr(mro[:1]), display=False)}]...")
        obj = BaseEstimator()
        self.assertTrue(hasattr(obj, "fit"))
        self.assertFalse(hasattr(obj, "transform"))
        self.assertFalse(hasattr(obj, "predict"))
        for mixin_klass in (ClassifierMixin, ClusterMixin, RegressorMixin,
                            TransformerMixin):
            _klass = type("Estimator", (mixin_klass, BaseEstimator), {})
            mro = _klass.__mro__
            print(f"[{print_multicolor(repr(mro[:3]), display=False)}]...")
            obj = _klass()
            if mixin_klass._estimator_type == "transformer":
                self.assertTrue(hasattr(obj, "fit"))
                self.assertTrue(hasattr(obj, "transform"))
                self.assertFalse(hasattr(obj, "predict"))
            else:
                self.assertTrue(hasattr(obj, "fit"))
                self.assertTrue(hasattr(obj, "predict"))
                self.assertFalse(hasattr(obj, "transform"))


if __name__ == "__main__":
    unittest.main()
