import torch.nn.functional as F
from torch import nn


class Offset0ModelMSE(nn.Module):
    """CEBRA model [1] with a single sample receptive field, without output normalization.
    
    [1] Learnable latent embeddings for joint behavioural and neural analysis, Schneider et al., Nature 2023
    """

    def __init__(self, num_input, n_embedding):
        """
        Args:
            num_input: The number of input dimensions. The tensor passed to
                the ``forward`` method will have shape ``(batch, num_input)``.
            n_embedding: The number of output dimensions. The tensor returned
                by the ``forward`` method will have shape ``(batch, n_embedding)``.

        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(
                num_input,
                n_embedding * 30,
            ),
            nn.GELU(),
            nn.Linear(n_embedding * 30, n_embedding * 30),
            nn.GELU(),
            nn.Linear(n_embedding * 30, n_embedding * 10),
            nn.GELU(),
            nn.Linear(int(n_embedding * 10), n_embedding)
        )

    def forward(self, inp):
        """Compute the embedding given the input signal.

            Args:
                inp: The input tensor of shape `num_samples x self.num_input x time`

            Returns:
                The output tensor of shape `num_samples x self.num_output x (time - receptive field)`.

            Based on the parameters used for initializing, the output embedding
            is normalized to the hypersphere (`normalize = True`).
        """
        return self.net(inp)

