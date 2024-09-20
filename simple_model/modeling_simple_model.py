"""The simple image classification model implemented in transformers using
pytorch backend.

This model classify a 28x28 black and white hand-written digits.
It is trained on the MNIST dataset.

Model Architecture:

img -> flatten -> linear -> sigmoid
"""

import torch
from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import v2 as transforms
from transformers import PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput

from .configuration_simple_model import SimpleModelConfig
from .pytorch_model import Model


class SimpleModel(PreTrainedModel):
    config_class = SimpleModelConfig
    
    def __init__(self, config: SimpleModelConfig):
        super().__init__(config)
        self.model = Model()
        
    def forward(self, image: Tensor, **kwargs) -> Tensor:
        return ImageClassifierOutput(logits=self.model(image))
