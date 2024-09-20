"""The simple image classification model implemented in pytorch.

This model classify a 28x28 black and white hand-written digits.
It is trained on the MNIST dataset.

Model Architecture:

img -> flatten -> linear -> sigmoid
"""
import torch.cuda
from torch import nn, Tensor


class Model(nn.Module):
    """Simple image classification model."""

    def __init__(self):
        super().__init__()
        self.lin0 = nn.Linear(28 * 28, 10)
        self.act = nn.Sigmoid()

    def forward(self, img: Tensor) -> Tensor:
        """Forward function."""
        x = img.flatten(start_dim=-3, end_dim=-1)
        x = self.lin0(x)
        x = self.act(x)
        return x

    @staticmethod
    def get_guess(x: Tensor) -> Tensor:
        return x.argmax(dim=-1)