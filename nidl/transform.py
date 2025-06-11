import numbers
import random
from abc import ABC
from abc import abstractmethod
from typing import Sequence, Union, Any, Dict, List

import numpy as np
import torch


TypeTransformInput = Union[np.ndarray, torch.Tensor, Dict]


class Transform(ABC):
    """Abstract class for all nidl transforms.

    When called, the input can be an instance of any type but we require output type to be np.ndarray
    All subclasses must overwrite :meth:`apply_transform` which takes an instance of :class:`~TypeTransformInput`,
    modifies it and returns the result.
    Args:
        p: Probability that this transform will be applied.
    """
    def __init__(self, p: float = 1.0):
        self.probability = self.parse_probability(p)
        # Whether or not we should parse the input data
        self.parse_input = True

    def __call__(self, data: TypeTransformInput) -> np.ndarray:
        """Transform data and return a result of type Numpy.
        Args:
            data: Instance of :class:`torch.Tensor` or :class:`numpy.ndarray`
        """
        # Some transforms such as Compose should not modify the input data
        if self.parse_input:
            subject = self.parse_data(data)
        else:
            subject = data

        if random.random() > self.probability:
            return subject

        with np.errstate(all='raise', under='ignore'):
            transformed = self.apply_transform(subject)
        return transformed

    def parse_data(self, data: TypeTransformInput) -> Any:
        return data

    def __repr__(self):
        if hasattr(self, 'args_names'):
            names = self.args_names
            args_strings = [f'{arg}={getattr(self, arg)}' for arg in names]
            if hasattr(self, 'invert_transform') and self.invert_transform:
                args_strings.append('invert=True')
            args_string = ', '.join(args_strings)
            return f'{self.name}({args_string})'
        else:
            return super().__repr__()

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def apply_transform(self, subject: Any) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def parse_probability(probability: float) -> float:
        is_number = isinstance(probability, numbers.Number)
        if not (is_number and 0 <= probability <= 1):
            message = (
                'Probability must be a number in [0, 1],'
                f' not {probability}'
            )
            raise ValueError(message)
        return probability


class Identity(Transform):

    def parse_data(self, data: TypeTransformInput) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Unexpected type: %s" % type(data))
        
    def apply_transform(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            raise ValueError("input type should be numpy array, got %s"%type(x))
        return x


class Compose(Transform):
    """Compose several transforms together.

    Args:
        transforms: Sequence of instances of
            :class:`Transform`.
        **kwargs: See :class:`Transform` for additional
            keyword arguments.

    """
    def __init__(self, transforms: Sequence[Transform], **kwargs):
        super().__init__(**kwargs)
        for transform in transforms:
            if not callable(transform):
                message = (
                    'One or more of the objects passed to the Compose'
                    f' transform are not callable: "{transform}"'
                )
                raise TypeError(message)
        self.transforms = list(transforms)
        self.parse_input = False

    def __len__(self):
        return len(self.transforms)

    def __getitem__(self, index) -> Transform:
        return self.transforms[index]

    def __repr__(self) -> str:
        return f'{self.name}({self.transforms})'

    def apply_transform(self, x: Any) -> np.ndarray:
        for transform in self.transforms:
            x = transform(x)
        return x


class MultiViewTransform(Transform):
    """Transform data into multiple views. 

    Args:
        transforms: Sequence of instances of :class:`Transform`. 
            Each transform creates a new view of the data.
        **kwargs: See :class:`Transform` for additional keyword arguments.
    """
    def __init__(self, transforms: Sequence[Transform], **kwargs):
        super().__init__(**kwargs)
        for transform in transforms:
            if not callable(transform):
                message = (
                    'One or more of the objects passed to the MultiViewTransform'
                    f' transform are not callable: "{transform}"'
                )
                raise TypeError(message)
        self.transforms = list(transforms)
        self.parse_input = False

    def __len__(self):
        return len(self.transforms)

    def __getitem__(self, index) -> Transform:
        return self.transforms[index]

    def __repr__(self) -> str:
        return f'{self.name}({self.transforms})'

    def apply_transform(self, x: Any) -> List[np.ndarray]:
        return [transform(x) for transform in self.transforms]
