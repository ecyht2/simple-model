"""The simple image classification model.

The model is implemented in different machine learning libraries.
This model classify a 28x28 black and white hand-written digits.
It is trained on the MNIST dataset.

Model Architecture:

img -> flatten -> linear -> sigmoid
"""

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel, 
    AutoModelForImageClassification, 
    AutoProcessor,
)

from .configuration_simple_model import SimpleModelConfig
from .modeling_simple_model import SimpleModel
from .processing_simple_model import SimpleModelProcessor
from .pytorch_model import Model as PytorchModel

AutoConfig.register("simple-model", SimpleModelConfig)
AutoModel.register(SimpleModelConfig, SimpleModel)
AutoModelForImageClassification.register(SimpleModelConfig, SimpleModel)
AutoProcessor.register("simple-model", SimpleModelProcessor)
AutoImageProcessor.register("simple-model", SimpleModelProcessor)

__all__ = ["PytorchModel", "SimpleModel", "SimpleModelConfig", "SimpleModelProcessor"]