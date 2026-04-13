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


@torch.no_grad()
def initialize_momentum_params(online_net: nn.Module, momentum_net: nn.Module):
    """Copies the parameters of the online network to the momentum network and
    deactivate its gradients computation.

    Parameters
    ----------
    online_net: nn.Module
        Online network (e.g. online backbone or online projection).

    momentum_net: nn.Module
        Momentum network (e.g. momentum backbone, momentum projection, etc...).
    """

    params_online = online_net.parameters()
    params_momentum = momentum_net.parameters()
    for po, pm in zip(params_online, params_momentum):
        pm.data.copy_(po.data)
        pm.requires_grad = False


class MomentumUpdater:
    def __init__(self, base_lambda: float = 0.996, final_lambda: float = 1.0):
        """Updates momentum parameters using exponential moving average.

        Parameters
        ----------
        base_lambda: float in [0,1], default=0.996
            Base value of the weight decrease coefficient.
        final_lambda: float in [0, 1], default=1.0
            Final value of the weight decrease coefficient.
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

        Parameters
        ----------
        online_net: nn.Module
            Online network (e.g. online backbone or online projection).
        momentum_net: nn.Module
            Momentum network (e.g. momentum backbone, or momentum projection).
        """

        for op, mp in zip(online_net.parameters(), momentum_net.parameters()):
            mp.data = (
                self.cur_lambda * mp.data + (1 - self.cur_lambda) * op.data
            )

    def update_lambda(self, cur_step: int, max_steps: int):
        """Computes the next value for the weighting decrease coefficient
        lambda using cosine annealing.

        Parameters
        ----------
        cur_step: int
            Number of gradient steps so far.
        max_steps: int
            Overall number of gradient steps in the whole training.
        """

        self.cur_lambda = (
            self.final_lambda
            - (self.final_lambda - self.base_lambda)
            * (math.cos(math.pi * cur_step / max_steps) + 1)
            / 2
        )
