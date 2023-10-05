"""
We will import this script for testing your models.
You have to write the function load_my_models(). It 
should take no arguments and return your final model.

The model should accept inputs matching your embedding
tensors, stored as embeddings.pth in the output.
The outputs should be of shape [10, 3, 96, 128].

Run this script with 'python validate_project2.py' to 
check that your implementation of load_my_models() works 
as intended.
"""

# Import modules for your own code
import torch
import torch.nn as nn
from fastai.learner import Learner

from src.config import IMG_SIZE
from src.ddpm import ConditionalDDPMCallback, EMA, ConditionalUnet
from src.loader import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Get your fine-tuned models here.
def load_my_models() -> nn.Module:
    """
    Loads your model.
    Use the imports to load the framework for your model.
    Then load the state dict such that your model is loaded
    with the correct weights.
    """
    dls = get_dataloader()

    # Create a new instance of the ConditionalDDPMCallback and load its state
    loaded_callback = ConditionalDDPMCallback(n_steps=0, beta_min=0, beta_max=0, cfg_scale=0)
    loaded_callback.load('output/callback_state.pth')

    # Instantiate model and learner
    model = ConditionalUnet(dim=IMG_SIZE, channels=3, num_classes=10).to(device)
    ddpm_learner = Learner(dls, model,
                           cbs=[loaded_callback,
                                EMA()],
                           loss_func=nn.L1Loss())

    # Load the Learner from the saved file
    ddpm_learner.load("learner_debug", device=None, with_opt=True, strict=True)

    return ddpm_learner


def test_load_my_models():
    final_model = load_my_models()
    final_model = final_model.to(device)

    # Send an example through the models, to check that they loaded properly
    images = torch.load('output/images.pth')
    embeddings = torch.load('output/embeddings.pth')

    pred_images, _ = final_model.conditional_ddpm.get_sample(
        (10, 3, IMG_SIZE, IMG_SIZE), torch.tensor(9).to(device), embeddings)

    return torch.allclose(pred_images, images, atol=1e-5)


def orig_test_load_my_models():
    # FIXME This was the original code
    final_model = load_my_models()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    final_model.eval()

    # Send an example through the models, to check that they loaded properly
    images = torch.load('output/images.pth')
    embeddings = torch.load('output/embeddings.pth')
    with torch.no_grad():
        pred_images = final_model(embeddings.to(device))

    return torch.allclose(pred_images, images, atol=1e-5)


if __name__ == '__main__':
    test_load_my_models()
