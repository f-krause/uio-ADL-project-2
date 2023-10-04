import os
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt

from ddpm import ConditionalDDPMCallback, EMA, ConditionalUnet


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


def run_ddpm_training(epochs=10, lr=3e-4, plot_loss=False):
    if torch.cuda.is_available():
        model = ConditionalUnet(dim=IMG_SIZE, channels=3, num_classes=10).cuda()
    else:
        model = ConditionalUnet(dim=IMG_SIZE, channels=3, num_classes=10)

    ddpm_learner = Learner(dls, model,
                           cbs=[ConditionalDDPMCallback(n_steps=1000, beta_min=0.0001, beta_max=0.02, cfg_scale=3),
                                EMA()],
                           loss_func=nn.L1Loss())

    # RUN TRAINING
    ddpm_learner.fit_one_cycle(epochs, lr, cbs=[SaveModelCallback(monitor="train_loss", fname="cifar10"),
                                                  WandbCallback(log_preds=False, log_model=True)])

    if plot_loss:
        ddpm_learner.recorder.plot_loss()
        plt.show()

