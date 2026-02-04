##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import math

import torch
from torch import nn


class MomentumUpdater:
    def __init__(self, base_lambda: float = 0.996, final_lambda: float = 1.0):
        """Updates momentum parameters using exponential moving average.

        Args:
            base_lambda (float, optional): base value of the weight decrease coefficient
                (should be in [0,1]). Defaults to 0.996.
            final_lambda (float, optional): final value of the weight decrease coefficient
                (should be in [0,1]). Defaults to 1.0.
        """

        super().__init__()

        assert 0 <= base_lambda <= 1
        assert 0 <= final_lambda <= 1 and base_lambda <= final_lambda

        self.base_lambda = base_lambda
        self.cur_lambda = base_lambda
        self.final_lambda = final_lambda

    @torch.no_grad()
    def update(self, online_net: nn.Module, momentum_net: nn.Module):
        """Performs the momentum update for each param group.

        Args:
            online_net (nn.Module): online network (e.g. online backbone,
            online projection, etc...).
            momentum_net (nn.Module): momentum network (e.g. momentum backbone,
                momentum projection, etc...).
        """

        for op, mp in zip(online_net.parameters(), momentum_net.parameters()):
            mp.data = (
                self.cur_lambda * mp.data + (1 - self.cur_lambda) * op.data
            )

    def update_lambda(self, cur_step: int, max_steps: int):
        """Computes the next value for the weighting decrease coefficient
        lambda using cosine annealing.

        Args:
            cur_step (int): number of gradient steps so far.
            max_steps (int): overall number of gradient steps in the whole training.
        """

        self.cur_lambda = (
            self.final_lambda
            - (self.final_lambda - self.base_lambda)
            * (math.cos(math.pi * cur_step / max_steps) + 1)
            / 2
        )
