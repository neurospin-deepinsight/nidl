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
from nidl.experiment import fetch_experiment
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

    def test_fetch_experiment(self):
        """ Test 'fetch_experiment'.
        """
        for expfile in self.configs:
            config = toml.load(expfile)
            project = config["project"]
            envs = config.get("environments")
            print(f"[{print_multicolor(project['name'], display=False)}] "
                  f"{project['desc'].lower()}...")
            exp = fetch_experiment(
                expfile, selector=envs.keys() if envs is not None else None,
                verbose=0)
            print(exp)

    def test_crossval_selection(self):
        """ Test the 'fetch_experiment' cv parameter.
        """
        expfile = os.path.join(self.dirname, "configs", "crossval.toml")
        config = toml.load(expfile)
        project = config["project"]
        envs = config.get("environments")
        print(f"[{print_multicolor(project['name'], display=False)}] "
              f"{project['desc'].lower()}...")
        exp = fetch_experiment(
            expfile, selector=envs.keys() if envs is not None else None,
            cv=["optimizer_0"], verbose=0)
        print(exp)


    def test_save_code(self):
        """ Test the 'fetch_experiment' save code option.
        """
        expfile = os.path.join(self.dirname, "configs", "crossval.toml")
        config = toml.load(expfile)
        project = config["project"]
        envs = config.get("environments")
        print(f"[{print_multicolor(project['name'], display=False)}] "
              f"{project['desc'].lower()}...")
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = fetch_experiment(
                expfile, selector=envs.keys() if envs is not None else None,
                cv=["optimizer_0", "training_2"], logdir=tmpdir, verbose=0)
        print(exp)


if __name__ == "__main__":
    unittest.main()
