"""Transformers configuration for Simple Model."""
from typing import Dict

from optimum.exporters.onnx.config import VisionOnnxConfig
from optimum.utils import NormalizedVisionConfig, DummyVisionInputGenerator
from transformers import PretrainedConfig

DummyVisionInputGenerator.SUPPORTED_INPUT_NAMES = (
    "pixel_values",
    "pixel_mask",
    "sample",
    "latent_sample",
    "image",
)


class SimpleModelConfig(PretrainedConfig):
    """Transformers configuration for Simple Model."""
    model_type = "simple-model"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.id2label = dict((i, str(i)) for i in range(10))


class SimpleModelOnnxConfig(VisionOnnxConfig):
    # Specifies how to normalize the BertConfig, this is needed to access common attributes
    # during dummy input generation.
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    # Sets the absolute tolerance to when validating the exported ONNX model against the
    # reference model.
    ATOL_FOR_VALIDATION = 1e-4

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "image": {
                0: "batch_size",
                1: "num_channels",
                2: "height",
                3: "width"
            },
        }
