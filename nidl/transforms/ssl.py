##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


""" Common transformations for self supervised learning.
"""


class ContrastiveTransformations:
    """ In self supervised learning, to allow efficient training, we need to
    prepare the data loading such that we sample two different, random
    augmentations for each image in the batch.
    The easiest way to do this is by creating a transformation that, when being
    called, applies a set of data augmentations to an image twice (or more).

    Parameters
    ----------
    transforms: transforms.Compose or equivalent
        the input set of transformations to be applied.
    n_views: int, default=2
        the number of positive examples.
    """
    def __init__(self, transforms, n_views=2):
        self.transforms = transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.transforms(x) for i in range(self.n_views)]
