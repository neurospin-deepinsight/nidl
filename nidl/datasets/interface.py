##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Interface datasets used as connectors.
"""

import torch
from torch.utils.data import (
    Dataset,
)


class HemisphereSplitDataset(Dataset):
    """
    A dataset that unpacks and splits hemisphere data across the channel
    dimension.

    Works with ``fs_xhemi`` style datasets where each item has shape
    ``(n_channels=n_modalities * 2, n_vertices)``. The first half of channels
    store all modalities for one hemisphere, and the second half for the other.
    The source dataset must provide ``get_fs_xhemi_feature_names`` returning
    channel names formatted as ``{lh|rh}.{modality}``.

    Parameters
    ----------
    dataset : Dataset
        Original dataset where each item returns a tensor of shape (2 * C, V)
        and optionally auxiliary variables.
    modalities : list[str] | None
        Names of returned modalities. If None, uses all available modalities.
        Default None.

    Attributes
    ----------
    source_dataset : Dataset
        The wrapped source dataset.
    n_subjects : int
        The total number of samples in the original dataset.
    channel_names : list[str] | None
        Names of all available channels from the source dataset.
    indices : list[list[int]]
        Indices for each hemisphere's selected modalities.

    Raises
    ------
    ValueError
        If the ``get_fs_xhemi_feature_names`` is not defined on the source
        dataset.
    """
    def __init__(
            self,
            source_dataset: Dataset,
            modalities: list[str] | None = None,
        ) -> None:
        if not hasattr(source_dataset, "get_fs_xhemi_feature_names"):
            raise ValueError(
                "The source dataset must provide a "
                "``get_fs_xhemi_feature_names`` method."
            )

        self.source_dataset = source_dataset
        self.n_subjects = len(source_dataset)
        self.channel_names = source_dataset.get_fs_xhemi_feature_names()
        self.hemisphere_indices = [
            [
                self.channel_names.index(f"{hemi}.{metric}")
                for metric in (modalities or self._extract_modalities())
            ]
            for hemi in ("lh", "rh")
        ]

    def _extract_modalities(self) -> list[str]:
        """
        Extract unique modality names from channel names.

        Returns
        -------
        list[str]
            List of unique modality names (without hemisphere prefix).
        """
        return list(
            {name.split(".")[1] for name in self.channel_names if "." in name}
        )

    def __len__(self) -> int:
        """
        Return the total number of samples: 2 * original dataset length.
        """
        return self.n_subjects * 2

    def __getitem__(
            self,
            index: int,
        ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        """
        Get item at given index, selecting appropriate hemisphere.

        Parameters
        ----------
        index : int
            Index of the item to retrieve.

        Returns
        -------
        tuple[torch.Tensor, ...] | torch.Tensor
            If source returns tuple, returns (hemisphere data, *rest).
            Otherwise returns just hemisphere data.
        """
        source_index, hemisphere_part = divmod(index, 2)
        batch = self.source_dataset[source_index]

        if isinstance(batch, (list, tuple)):
            data, *rest = batch
            return (data[self.hemisphere_indices[hemisphere_part]], *rest)

        return batch[self.hemisphere_indices[hemisphere_part]]
