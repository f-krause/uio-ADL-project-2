# TODO adapt to new case
import torchvision
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import litdata


class ToRGBTensor:
    """Code from Mariuaas copied from Discourse"""
    def __call__(self, img):
        return transforms.functional.to_tensor(img).expand(3, -1, -1)  # Expand to 3 channels

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
    
class GetMoiraLabel:
    def __call__(self, tensor):
        return int(tensor.squeeze()[1])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def get_data(input_shape: list):
    """
    Get train and test data loader based on educload .tar data
    Code adapted from Mariuaas from Discourse
    """
    # Define data path
    DATA_PATH = '/projects/ec232/data/'

    # Define mean and std from ImageNet data
    IN_MEAN = [0.485, 0.456, 0.406]
    IN_STD = [0.229, 0.224, 0.225]

    # Define postprocessing / transform of data modalities
    postprocess = (  # Create tuple for image and class...
        transforms.Compose([  # Handles processing of the .jpg image
            ToRGBTensor(),  # Convert from PIL image to RGB torch.Tensor
            transforms.Resize(input_shape[:2]),  # Resize images
            transforms.Normalize(IN_MEAN, IN_STD),  # Normalize image to correct mean/std
        ]),
        transforms.Compose([  # Handles processing of the .jpg image
            transforms.ToTensor(), # Convert .scores.npy file to tensor
            GetMoiraLabel(),
        ])
    )

    # Load data
    data = litdata.LITDataset(
        "CarRecs",
        DATA_PATH,
        override_extensions=["jpg", "scores.npy"],  # first load image, then scores
    ).map_tuple(*postprocess)

    return data
