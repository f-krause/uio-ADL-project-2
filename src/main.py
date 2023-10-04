# TODO adapt to new case
import argparse
from datetime import datetime
from tqdm import tqdm
import os

from get_data import get_data
from train import run_vae_training


# Define some flags to specify parameters of training in shell
parser = argparse.ArgumentParser(description="Specify parameters of model training")
parser.add_argument("-v", "--vae", action="store_true", help="run LoRA based training")
parser.add_argument("-s", "--save", action="store_true", help="save model in output directory")
parser.add_argument("-e", "--epochs", default=5, type=int, help="specify number of epochs")
args = parser.parse_args()


def main():
    # Specify data and training parameters
    INPUT_SHAPE = [96, 128, 3]  # TODO cross-check image dimensions!
    BATCH_SIZE = 64

    # Load data from educloud server if data path exists
    print(datetime.now(), "Loading data from educloud server")
    train_data = get_data(INPUT_SHAPE)

    if args.vae:
        # Run conditional VAE training
        print(datetime.now(), f"Starting VAE training with {args.epochs} epochs")
        run_vae_training(INPUT_SHAPE, train_data, args.epochs, BATCH_SIZE, args.save)
        
    else:
        # TODO ADD OTHER APPROACH
        # print(datetime.now(), f"Starting LoRA training with {args.epochs} epochs and r={args.rank}")
        # run_lora_tuning(train_loader, test_loader, args.epochs, args.rank, args.save)
        pass


if __name__ == '__main__':
    main()
