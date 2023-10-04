# TODO adapt to new case
# TODO include seed!
import os.path
import torch
import time
import timm
from timm import optim
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F

# import our models
from cVAE import CVAE

# Automatically define torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def measure(fnc):
    """Measure helper function to measure training time"""
    start = time.time()
    losses = fnc()
    end = time.time()
    print(f'Time taken: {end - start}')
    return losses


def evaluate(model, dataloader, loss_fnc):
    """TODO How to evaluate?"""
    pass


def check_save_path(path):
    if os.path.isfile(path):
        path, file_type = path.split(".")
        r_time = str(datetime.now())[-5:]
        return path + "_" + r_time + "." + file_type
    return path


def save_data(model_name, model, losses):
    """Helper function to save models, losses and images. Prevents file overwriting"""

    # Save model
    save_path_model = check_save_path(f"output/{model_name}_model.pth")
    print(datetime.now(), f"Saving model to {save_path_model}")
    torch.save(model.state_dict(), save_path_model)  # save state dict of model

    # Save losses
    save_path_csv = check_save_path(f"output/losses_{model_name}_model.csv")
    np.savetxt(save_path_csv, np.array(losses), delimiter=",")  # save losses as csv

    # Save images
    for i in range(10):
        save_path_img = check_save_path(f"output/{model_name}_img{i}.jpg")
        # TODO save images


def get_vae_loss(x, x_hat, mean, log_var):
    """
    Inputs:
        x       : [torch.tensor] Original sample
        x_hat   : [torch.tensor] Reproduced sample
        mean    : [torch.tensor] Mean mu of the variational posterior given sample x
        log_var  : [torch.tensor] log of the variance sigma^2 of the variational posterior given sample x
    """

    # Reconstruction loss
    reproduction_loss = ((x - x_hat)**2).sum()

    # KL divergence
    kl_divergence = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    # Get the total loss
    loss = reproduction_loss + kl_divergence

    return loss


def run_vae_training(input_shape, train_data, epochs, batch_size, save_model: bool = False):
    """Run conditional VAE training based on exercise sheet 4"""
    Z_DIM = 3
    conditioned_model = CVAE(Z_DIM, n_classes=10, n_channels=3, img_size=input_shape[:2]).to(device)

    # Feel free to tweak the training parameters
    lr = 0.01
    optimizer = Adam(conditioned_model.parameters(), lr=lr)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    # Train for a few epochs
    conditioned_model.train()

    losses = []
    # TODO USE: with traindata.shufflecontext(): 2 # do training loop ...
    for epoch in range(epochs):
        train_bar = tqdm(iterable=train_loader)
        for i, (x, c) in enumerate(train_bar):
            total_loss = 0
            # Get x_hat, mean, log_var,and cls_token from the conditioned_model
            x, x_hat, mean, log_var, cls_token = conditioned_model.forward(x, c)

            # Get vae loss
            vae_loss = get_vae_loss(x, x_hat, mean, log_var)

            # Get cross entropy loss for the cls token
            cls_loss = F.cross_entropy(cls_token, F.one_hot(c, num_classes=10).double(), reduction='sum')

            # Add the losses as a weighted sum. NB: We weight the cls_loss by 10 here, but feel free to tweak it.
            loss = vae_loss + cls_loss * 10
            total_loss += loss.item()
            losses.append(total_loss)

            # Update model parameters based on loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_bar.set_description(f'Epoch [{epoch+1}/{epochs}]')
            train_bar.set_postfix(loss=loss.item() / len(x))

    # losses = measure(lambda: train(model, train_data, epochs, optimizer))  # run model training

    if save_model:
        save_data("cVAE_first", conditioned_model, losses)


# def train(model, train_data, epochs, optimizer):
#     """Train model on train data for given optimizer and loss function"""
#     model.train()
#     losses = []
#     with train_data.shufflecontext():
#         for epoch in range(epochs):
#             total_loss = 0
#             for batch in tqdm(train_data):
#                 optimizer.zero_grad()
#
#                 # Load data onto device
#                 inputs, targets = batch
#                 inputs = inputs.to(device)
#                 targets = targets.to(device)
#
#                 # Predictions
#                 outputs = model(inputs)
#                 loss = loss_fnc(outputs, targets)
#
#                 # Optimize
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()
#             losses.append(total_loss)
#             print(datetime.now(), f'Epoch: {epoch + 1}, loss: {total_loss}')
#
#     return losses
