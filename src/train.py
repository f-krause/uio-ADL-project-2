import os
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from fastai.learner import Learner
from torchvision.utils import save_image
import torch.nn.functional as F

from ddpm import ConditionalDDPMCallback, EMA, ConditionalUnet
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_save_path(path):
    if os.path.isfile(path):
        path, file_type = path.split(".")
        r_time = str(datetime.now())[-5:]
        return path + "_" + r_time + "." + file_type
    return path


def save_learner(model_name, learner, losses):
    """Helper function to save models, losses and images. Prevents file overwriting"""

    # Save model
    save_path_model = check_save_path(f"output/{model_name}_model.pth")
    print(datetime.now(), f"Saving model to {save_path_model}")
    learner.save("learner_debug")

    callback = learner.conditional_ddpm
    callback.save('models/callback_state.pth')

    # Save losses
    # save_path_csv = check_save_path(f"output/losses_{model_name}_model.csv")
    # np.savetxt(save_path_csv, np.array(losses), delimiter=",")  # save losses as csv


def generate_images(learner, k=10, store_jpg=True, store_tensors=True):
    # Save images
    pred_imgs, embeddings = learner.conditional_ddpm.get_sample((k, 3, IMG_SIZE, IMG_SIZE), torch.tensor(9).to(device))

    pred_imgs = F.interpolate(pred_imgs, size=(96, 128), mode='bilinear', align_corners=False)  # upsample images

    if store_tensors:
        torch.save(pred_imgs, "../output/images.pth")
        torch.save(embeddings)

    if store_jpg:
        for i, pred_img in enumerate(pred_imgs):
            save_path_img = check_save_path(f"../output/img_{i}.jpg")
            save_image(pred_img, save_path_img)

    return pred_imgs


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
        save_learner("WIP_ddpm", model, losses)
