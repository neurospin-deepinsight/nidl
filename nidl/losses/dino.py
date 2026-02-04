from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as func
from torch import Tensor
from torch.nn import Module, Parameter


class DINOLoss(Module):
    """Implementation of the DINO loss [1]_.

    This implementation follows the code published by the authors:
    https://github.com/facebookresearch/dino

    It supports global and local image crops. A linear warmup schedule for the
    teacher temperature is implemented to stabilize training at the beginning.
    Centering is applied to the teacher output to avoid model collapse.

    Parameters
    ----------
    output_dim: int, default=4096
        Dimension of the model output.
    warmup_teacher_temp: float, default=0.04
        Initial temperature for the teacher network.
    teacher_temp: float, default=0.07
        Final temperature for the teacher network.
    warmup_teacher_temp_epochs: int, default=30
        Number of epochs for the warmup phase of the teacher temperature.
    student_temp: float, default=0.1
        Temperature for the student network.
    center_momentum: float, default=0.9
        Momentum term for the center calculation.

    Examples
    --------
    >>> # initialize loss function
    >>> loss_fn = DINOLoss(128)
    >>>
    >>> # generate a view of the images with a random transform
    >>> view = transform(images)
    >>>
    >>> # embed the view with a student and teacher model
    >>> teacher_out = teacher(view[:2])
    >>> student_out = student(view[2:])
    >>>
    >>> # calculate loss
    >>> loss = loss_fn(teacher_out, student_out)

    References
    ----------
    .. [1] Caron, M., et al., "Emerging Properties in Self-Supervised Vision
           Transformers." ICCV, 2021. https://arxiv.org/abs/2104.14294

    """

    def __init__(
        self,
        output_dim: int = 4096,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.07,
        warmup_teacher_temp_epochs: int = 30,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        """Initializes the DINOLoss Module.

        Args:
            warmup_teacher_temp:
                Initial temperature for the teacher network.
            warmup_teacher_temp_epochs:
                Number of epochs for the warmup phase of the teacher
                temperature.
        """
        super().__init__()

        self.teacher_temp = teacher_temp
        self.student_temp = student_temp

        self.center: Parameter
        self.register_buffer("center", torch.zeros(1, 1, output_dim))
        self.center_momentum = center_momentum

        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.teacher_temp_schedule = torch.linspace(
            start=warmup_teacher_temp,
            end=teacher_temp,
            steps=warmup_teacher_temp_epochs,
        )

    def forward(
        self,
        teacher_out: Tensor,
        student_out: Tensor,
        epoch: int | None = None,
    ) -> Tensor:
        """Cross-entropy between softmax outputs of the centered teacher and
        student.

        Parameters
        ----------
        teacher_out: torch.Tensor of shape (n_views, batch_size, n_features)
            Features from the teacher model. Each tensor represents one
            (global) view of the batch.
        student_out: torch.Tensor of shape (n_views, batch_size, n_features)
            Features from the student model. Each tensor represents one
            (local) view of the batch.
        epoch: int or None
            The current epoch used to set the teacher temperature.
            If None, the default `teacher_temp` is used.

        Returns
        -------
        loss: torch.Tensor
            The average cross-entropy loss.
        """

        # Get teacher temperature
        if epoch is not None:
            if epoch < self.warmup_teacher_temp_epochs:
                teacher_temperature = self.teacher_temp_schedule[epoch]
            else:
                teacher_temperature = torch.tensor(self.teacher_temp)
        else:
            teacher_temperature = torch.tensor(self.teacher_temp)

        # Calculate cross-entropy loss.
        t_out = func.softmax(
            (teacher_out - self.center) / teacher_temperature, dim=-1
        )
        s_out = func.log_softmax(student_out / self.student_temp, dim=-1)

        # Calculate feature similarities, ignoring the diagonal
        # b = batch_size
        # t = n_views_teacher
        # s = n_views_student
        # d = n_features
        loss = -torch.einsum("tbd,sbd->ts", t_out, s_out)
        loss.fill_diagonal_(0)

        # Number of loss terms, ignoring the diagonal
        n_terms = loss.numel() - loss.diagonal().numel()
        batch_size = teacher_out.shape[1]

        loss = loss.sum() / (n_terms * batch_size)

        # Update the center used for the teacher output
        self.update_center(teacher_out)

        return loss

    @torch.no_grad()
    def update_center(self, teacher_out: Tensor) -> None:
        """Moving average update of the center used for the teacher output.

        Parameters
        ----------
        teacher_out: torch.Tensor of shape (n_views, batch_size, n_features)
            Features from the teacher model.
        """

        # Calculate the batch center using the specified center function
        batch_center = torch.mean(teacher_out, dim=(0, 1), keepdim=True)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()

        # Update the center with a moving average
        self.center.data = (
            self.center * self.center_momentum
            + batch_center * (1 - self.center_momentum)
        )
