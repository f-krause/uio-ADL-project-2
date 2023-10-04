# Parts adapted from: https://nghiaho.com/?p=2741
import os
import glob
import numpy as np
from PIL import Image
from config import *
from datetime import datetime
from fastai.data import *

if DATA_PATH:
    PATH = DATA_PATH
else:
    PATH = "/fp/homes01/u01/ec-felixek/pc/Documents/code/data/Dataset"
    if not os.path.isdir(PATH):
        print(datetime.now(), "Loading data locally")
        PATH = "data"
    print(datetime.now(), "Loading data from educloud server")


def load_dataset():
    types = ['*.jpg', '*.npy']
    images, labels = [], []
    for data_type in types:
        for file in glob.glob(f"{PATH}/{data_type}"):
            if data_type == '*.jpg':
                image = Image.open(file)
                image = image.resize((IMG_SIZE, IMG_SIZE))
                image = np.array(image, dtype=np.uint8)
                image = image if len(image.shape) == 3 else np.dstack([image] * 3)
                images.append(image)
            if data_type == '*.npy':
                label = np.load(file)
                labels.append(label)

    if not images:
        raise Exception("No data loaded!")
    labels = np.array(labels)[:, :, 1].squeeze()  # only keep label of Moira
    return np.array(images), labels


def get_dataloader(images, labels):
    def pass_index(idx):
        return idx

    def get_x(i):
        return images[i]

    def get_y(i):
        return labels[i]

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=pass_index,
        get_x=get_x,
        get_y=get_y,
        batch_tfms=Normalize.from_stats(0.5, 0.5))

    # pass in a list of index
    dls = dblock.dataloaders(list(range(images.shape[0])), bs=BATCH_SIZE)

    return dls
