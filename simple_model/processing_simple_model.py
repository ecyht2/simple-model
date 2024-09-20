"""Transformers processing for Simple Model."""
import torch
from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import v2 as transforms
from transformers import BaseImageProcessor, BatchFeature


class SimpleModelProcessor(BaseImageProcessor):
    """Transformers processing for Simple Model."""
    model_input_names = ["image"]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size = {"height": 28, "width": 28}
        self.image_mean = [
            0.1307,
            0.1307,
            0.1307
        ]
        self.image_std = [
            0.3081,
            0.3081,
            0.3081
        ]
    
    def preprocess(self, image: Image | list[Image], **kwargs) -> Tensor:
        """Preprocess the image."""
        if isinstance(image, list):
            image = list(map(lambda img: img.convert("L"), image))
        else:
            image = image.convert("L")

        transform_image = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        image_tensor = transform_image(image)
        if isinstance(image, list):
            image_tensor = torch.stack(image_tensor, dim=0)
        else:
            image_tensor = image_tensor.unsqueeze(0)
        return BatchFeature(data={"image": image_tensor}, tensor_type="pt")
