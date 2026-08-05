##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import nidl
import doctest
import importlib
import pkgutil
import unittest


def load_tests(loader, tests, ignore):

    for _, module_name, ispkg in pkgutil.walk_packages(
            nidl.__path__,
            nidl.__name__ + "."):
        if not module_name.startswith("nidl.surface"):
            continue
        print(f"-> testing: {module_name}...")
        module = importlib.import_module(module_name)
        tests.addTests(
            doctest.DocTestSuite(
                module,
                optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
            )
        )
    return tests


if __name__ == "__main__":
    unittest.main()
