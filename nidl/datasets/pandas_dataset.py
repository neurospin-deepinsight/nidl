##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from typing import Any, Callable, Optional, Union

import nibabel
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def default_loader(path: str) -> Any:
    """Default image loader function."""
    if path.endswith((".nii", ".nii.gz")):
        return nibabel.load(path).get_fdata()
    elif path.endswith((".npy", ".npz")):
        return np.load(path)
    elif path.lower().endswith(
        (
            ".jpg",
            ".jpeg",
            ".png",
            ".ppm",
            ".bmp",
            ".pgm",
            ".tif",
            ".tiff",
            ".webp",
        )
    ):
        return Image.open(path).convert("RGB")
    else:
        raise ValueError(f"Unsupported file type: {path}")


class ImageDataFrameDataset(Dataset):
    """PyTorch Dataset for loading images from a pandas DataFrame.

    This dataset assumes that the DataFrame contains:

    - one column with file paths to image data;
    - zero or more additional columns containing target labels (optional).

    Images are loaded on-the-fly from disk when accessed. Labels (if provided)
    are extracted from the specified column(s) and returned alongside the
    image.

    Parameters
    ----------
    df: pd.DataFrame or pd.Series or str
        DataFrame containing image paths and optional labels:

        - if a DataFrame, it should contain at least one column with the
          image paths;
        - if a Series, it should contain the image paths;
        - if str, it should be the path to a CSV file.
    image_col: str, default="image_path"
        Name of the column in `df` containing image file paths.
    label_cols: str, list of str, default=None
       Name of the column(s) containing label(s):

       - if None (default), no labels are returned.
       - if string, it should be the name of a single column in `df`.
       - if list, it should contain the names of multiple columns in `df`.
    transform: Callable, default=None
        Optional transform that takes in the loaded image and returns a
        transformed version.
    target_transform: Callable, Dict[str, Callable], default=None
        Optional transform applied to the label(s):

        - if callable: applied to all labels, e.g. `lambda y: torch.tensor(y)`
        - if dictionary: apply different transforms per column. In that case,
          the keys should be included in the column names in `label_cols` and
          values must be callable.
    return_none_if_no_label: bool, default=True
        If True, returns `(<img>, None)` when getting an item and `label_cols`
        is empty or None (default). Otherwise, only `<img>` is returned.
    image_loader: Callable, default=default_loader
        Function to load the image from the file path. It takes a string (the
        file path) as input and returns the loaded image. By default, it
        accepts the following:

        - all image extensions supported by PIL (e.g., .jpg, .png, .bmp etc.)
        - numpy arrays (e.g., .npy, .npz)
        - 3D medical images (e.g., .nii, .nii.gz) using nibabel
    is_valid_file: Callable, default=None
        Function to check if a file path (string) is a valid image file (e.g.
        correct extension). If None (default), the file extension is checked
        against a list of known image extensions. Invalid files will be
        filtered out from the dataset.
    is_valid_label: Callable, Dict[str, Callable], default=None
        Function to check if a label is valid. If None (default), all labels
        are considered valid. This can be used to filter out samples with
        invalid labels from the dataset, e.g. NaN. If `label_cols` is a string,
        it takes a label as input and returns a boolean. If `label_cols` is a
        list, it takes a list of labels as input and returns a boolean.
    read_csv_kwargs: Optional[dict], default=None
        Additional keyword arguments to pass to `pd.read_csv` if `df` is a
        string path. For instance you can define the proper '\t' separator
        when working with a TSV file.

    Attributes
    ----------
    df: pd.DataFrame
        The DataFrame containing image paths and labels.
    imgs: list
        List of image paths (before loading).
    targets: list
        List of labels (before any transformations).

    Examples
    --------
    Dataset for supervised computer vision tasks:
    >>> import pandas as pd
    >>> from datasets.pandas_dataset import ImageDataFrameDataset
    >>> df = pd.DataFrame({
    ...     'image_path': ['image1.jpg', 'image2.jpg'],
    ...     'label': ['cat', 'dog']
    ... })
    >>> dataset = ImageDataFrameDataset(df, image_col='image_path',
    ...     label_cols='label')
    >>> image, label = dataset[0]
    >>> print(label)
    "cat"
    >>> print(type(image))
    <class 'PIL.Image.Image'>

    Dataset for unsupervised computer vision tasks:
    >>> import pandas as pd
    >>> from datasets.pandas_dataset import ImageDataFrameDataset
    >>> df = pd.DataFrame({
    ...     'image_path': ['image1.jpg', 'image2.jpg']
    ... })
    >>> dataset = ImageDataFrameDataset(df, image_col='image_path')
    >>> image, _ = dataset[0]
    >>> print(type(image))
    <class 'PIL.Image.Image'>

    Dataset for 3D medical images:
    >>> df = pd.DataFrame({
    ...     'image_path': ['mri1.nii', 'mri2.nii'],
    ...     'diagnosis': ['patient', 'control'],
    ...     'age': [30, 25]
    ... })
    >>> target_transform = {"diagnosis": lambda x: 1 if x == 'patient' else 0}
    >>> dataset = ImageDataFrameDataset(
    ...     df, image_col='image_path',
    ...     label_cols=['diagnosis', 'age'],
    ...     target_transform=target_transform
    ... )
    >>> image_mri, (label, age) = dataset[0]
    >>> print(label_mri, age_mri)
    (30, 1)
    >>> print(type(image_mri))
    <class 'nibabel.nifti1.Nifti1Image'>

    """

    IMG_EXTENSIONS = (
        ".jpg",
        ".jpeg",
        ".png",
        ".ppm",
        ".bmp",
        ".pgm",
        ".tif",
        ".tiff",
        ".webp",
        ".npz",
        ".npy",
        ".nii",
        ".nii.gz",
    )

    def __init__(
        self,
        df: Union[pd.DataFrame, pd.Series, str],
        image_col: str = "image_path",
        label_cols: Optional[Union[str, list[str]]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[
            Union[Callable, dict[str, Callable]]
        ] = None,
        return_none_if_no_label: bool = True,
        image_loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        is_valid_label: Optional[Callable[[Any], bool]] = None,
        read_csv_kwargs: Optional[dict] = None,
    ):
        if isinstance(df, str):
            read_csv_kwargs = read_csv_kwargs or {}
            df = pd.read_csv(df, **read_csv_kwargs)
        elif isinstance(df, pd.Series):
            df = df.to_frame(name=image_col)

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a DataFrame, Series, or path to CSV.")

        if image_col not in df.columns:
            raise ValueError(f"{image_col} column not found in DataFrame.")

        self.df = df.copy()
        self.image_col = image_col
        self.label_cols = self._parse_labels(df, label_cols)
        self.transform = transform
        self.target_transform = target_transform
        self.return_none_if_no_label = return_none_if_no_label
        self.image_loader = image_loader

        if is_valid_file is None:
            # Default to checking file extensions
            def is_valid_file(file_path: str) -> bool:
                return file_path.lower().endswith(self.IMG_EXTENSIONS)

        self.is_valid_file = is_valid_file

        if is_valid_label is None:
            # Default to considering all labels valid
            def is_valid_label(label: Any) -> bool:
                return True

        self.is_valid_label = is_valid_label

        # Filter valid files
        mask = self.df[image_col].apply(self.is_valid_file)
        if mask.sum() < len(self.df):
            print(
                f"Warning: {len(self.df) - mask.sum()} invalid files found. "
                "These will be removed from the dataset."
            )
            self.df = self.df[mask]

        # Filter valid labels
        if self.label_cols is not None:
            if isinstance(self.label_cols, str):
                mask = self.df[self.label_cols].apply(is_valid_label)
            else:  # Multiple label columns: apply to each row
                mask = self.df[self.label_cols].apply(is_valid_label, axis=1)
            if mask.sum() < len(self.df):
                print(
                    f"Warning: {len(self.df) - mask.sum()} samples with "
                    "invalid labels found. These will be removed from the "
                    "dataset."
                )
                self.df = self.df[mask]

        self.targets = self._extract_targets()
        self.imgs = self.df[image_col].tolist()

        assert len(self.imgs) == len(self.targets)

    def _extract_targets(self):
        if self.label_cols is None:
            return [None] * len(self.df)
        elif isinstance(self.label_cols, str):
            return self.df[self.label_cols].tolist()
        else:
            return self.df[self.label_cols].values.tolist()

    def _parse_labels(
        self, df: pd.DataFrame, label_cols: Optional[Union[str, list[str]]]
    ):
        if label_cols is None:
            return None
        elif isinstance(label_cols, str) and label_cols not in df.columns:
            raise ValueError(f"Column '{label_cols}' not found in DataFrame.")
        elif (
            isinstance(label_cols, list)
            and len(set(label_cols) - set(df.columns)) > 0
        ):
            raise ValueError(
                f"Columns {set(label_cols) - set(df.columns)} "
                "not found in DataFrame."
            )
        elif not (isinstance(label_cols, (str, list))):
            raise TypeError(
                f"label_cols must be a string or a list of strings, got "
                f"{label_cols}."
            )
        return label_cols

    def apply_transform(self, image):
        """Apply the specified transform to the image."""
        if self.transform is not None:
            return self.transform(image)
        return image

    def apply_target_transform(self, label):
        """Apply the specified target transform to the label(s)."""
        if self.target_transform is None:
            return label

        if isinstance(self.target_transform, dict):
            if isinstance(label, (list, tuple)):
                return [
                    self.target_transform.get(col, lambda x: x)(val)
                    for col, val in zip(self.label_cols, label)
                ]
            else:
                return self.target_transform.get(self.label_cols, lambda x: x)(
                    label
                )

        return self.target_transform(label)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.imgs[idx]
        image = self.image_loader(image_path)
        label = self.targets[idx]

        label = self.apply_target_transform(label)
        image = self.apply_transform(image)

        if not self.return_none_if_no_label and (
            self.label_cols is None
            or (
                isinstance(self.label_cols, list) and len(self.label_cols) == 0
            )
        ):
            return image

        return image, label
