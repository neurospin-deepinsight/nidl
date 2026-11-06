##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Dataset related tools.
"""

import torch
from torch.utils.data import (
    DataLoader,
)


def get_channel_stats(
        dataloader: DataLoader
    ) -> tuple[list[float], list[float]]:
    """
    Calculates the per-channel mean and standard deviation of a dataset.

    This function iterates through a PyTorch DataLoader to compute the global
    mean and standard deviation for each channel. It is designed to handle
    large datasets efficiently without loading all samples into memory at once.

    Parameters
    ----------
    dataloader : DataLoader
        The DataLoader yielding batches of data.
        Data are expected to be PyTorch tensors of shape
        `(Batch, Channels, *)` and scaled to floating-point.

    Returns
    -------
    mean : list[float]
        The calculated mean value for each channel.
    std : list[float]
        The calculated standard deviation value for each channel.

    Raises
    ------
    ValueError
        If the data have wrong shape.
    """
    data = next(iter(dataloader))
    if data.ndim < 3:
        raise ValueError(
            "Data are expected to be PyTorch tensors of shape "
            f"`(Batch, Channels, *)`: {data.shape}"
        )
    n_channels = data.shape[1]
    dims = [idx for idx in range(data.ndim) if idx != 1]

    channels_sum = torch.zeros(n_channels)
    channels_squared_sum = torch.zeros(n_channels)
    for data in dataloader:
        channels_sum += torch.mean(data, dim=dims)
        channels_squared_sum += torch.mean(data ** 2, dim=dims)
 
    mean_tensor = channels_sum / len(dataloader)
    std_tensor = torch.sqrt(
        (channels_squared_sum / len(dataloader)) -
        (mean_tensor ** 2)
    )
    
    return mean_tensor.tolist(), std_tensor.tolist()
