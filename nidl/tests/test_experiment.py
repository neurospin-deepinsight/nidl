##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


import os
import unittest
from nidl.experiment import fetch_experiment


class TestExperiment(unittest.TestCase):
    """ Test experiement.
    """
    def setUp(self):
        """ Setup test.
        """
        dirname = os.path.abspath(os.path.dirname(__file__))
        self.expfile = os.path.join(dirname, "experiment.toml")

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_fetch_experiment(self):
        """ Test 'fetch_experiment'.
        """
        exp = fetch_experiment(self.expfile, selector=["tiny", "tl"],
                               verbose=1)


if __name__ == "__main__":
    unittest.main()
