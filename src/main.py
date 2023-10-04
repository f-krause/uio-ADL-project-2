import argparse
from datetime import datetime

from config import *
from loader import load_dataset, get_dataloader
from train import run_ddpm_training


def main():
    if MODEL == "DDPM":
        images, labels = load_dataset()
        print(images)
        print(labels)
        dls = get_dataloader(images, labels)

        # Run DDPM training
        print(datetime.now(), f"Starting DDPM training with {EPOCHS} epochs")
        run_ddpm_training(dls)

    else:
        # MAYBE ANOTHER APPROACH
        # print(datetime.now(), f"Starting LoRA training with {args.epochs} epochs and r={args.rank}")
        # run_lora_tuning(train_loader, test_loader, args.epochs, args.rank, args.save)
        raise NotImplementedError()


if __name__ == '__main__':
    main()
