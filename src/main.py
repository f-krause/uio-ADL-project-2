import argparse
from datetime import datetime

from config import *
from loader import get_dataloader
from train import run_ddpm_training


def main():
    if MODEL == "DDPM":
        dls = get_dataloader()

        # Run DDPM training
        print(datetime.now(), f"Starting DDPM training with {EPOCHS} epochs")
        run_ddpm_training(dls)

    else:
        # MAYBE ANOTHER APPROACH
        raise NotImplementedError()


if __name__ == '__main__':
    main()
