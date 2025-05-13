##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


import glob
import os
import toml
import unittest
from nidl.experiment import fetch_experiment
from nidl.utils import print_multicolor


class TestExperiment(unittest.TestCase):
    """ Test experiement.
    """
    def setUp(self):
        """ Setup test.
        """
        dirname = os.path.abspath(os.path.dirname(__file__))
        self.configs = glob.glob(os.path.join(dirname, "configs", "*.toml"))
        print(self.configs)

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_fetch_experiment(self):
        """ Test 'fetch_experiment'.
        """
        for expfile in self.configs:
            config = toml.load(expfile)
            project = config["project"]
            envs = config["environments"]
            print(f"[{print_multicolor(project['name'], display=False)}] "
                  f"{project['desc'].lower()}...")
            exp = fetch_experiment(expfile, selector=envs.keys(),
                                   verbose=0)
            print(exp)


if __name__ == "__main__":
    unittest.main()
