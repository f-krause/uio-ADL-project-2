import os
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from fastai.learner import Learner


from ddpm import ConditionalDDPMCallback, EMA, ConditionalUnet
from config import *


def check_save_path(path):
    if os.path.isfile(path):
        path, file_type = path.split(".")
        r_time = str(datetime.now())[-5:]
        return path + "_" + r_time + "." + file_type
    return path


def save_data(model_name, learner, losses):
    """Helper function to save models, losses and images. Prevents file overwriting"""

    # Save model
    save_path_model = check_save_path(f"output/{model_name}_model.pth")
    print(datetime.now(), f"Saving model to {save_path_model}")
    learner.save(save_path_model)
    # torch.save(model.state_dict(), save_path_model)  # save state dict of model

    # Save losses
    # save_path_csv = check_save_path(f"output/losses_{model_name}_model.csv")
    # np.savetxt(save_path_csv, np.array(losses), delimiter=",")  # save losses as csv

    # Save images
    for i in range(10):
        learner.forward()
        save_path_img = check_save_path(f"output/{model_name}_img{i}.jpg")
        # TODO save images


def run_ddpm_training(dls, plot_loss=False):
    # Partly from https://github.com/fastai/fastdiffusion/blob/master/nbs/tcapelle/Diffusion_models_with_fastai_conditional_cifart_EMA.ipynb
    if torch.cuda.is_available():
        model = ConditionalUnet(dim=IMG_SIZE, channels=3, num_classes=10).cuda()
    else:
        model = ConditionalUnet(dim=IMG_SIZE, channels=3, num_classes=10)

    ddpm_learner = Learner(dls, model,
                           cbs=[ConditionalDDPMCallback(n_steps=1000, beta_min=0.0001, beta_max=0.02, cfg_scale=3),
                                EMA()],
                           loss_func=torch.nn.L1Loss())

    # Automatically find good learning rate
    # ddpm_learner.lr_find()

    # RUN TRAINING
    ddpm_learner.fit_one_cycle(EPOCHS, LR)

    losses = []  # TODO

    if plot_loss:
        ddpm_learner.recorder.plot_loss()
        plt.show()

    if SAVE_MODEL:
        save_data("WIP_ddpm", model, losses)
