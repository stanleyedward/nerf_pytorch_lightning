import torch
from torch import nn
import matplotlib.pyplot as plt
from typing import List
from tqdm import tqdm
import json
from pathlib import Path
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from dataset import LegoDataModule
from model import NeRFLightning
from rendering import rendering
from model import Sphere
from config import (
    ORIGIN,
    RADIUS,
    HEIGHT,
    WIDTH,
    SAVE_DIR,
    DATA_DIR,
    IMG_SIZE,
    LEARNING_RATE,
    TN,
    TF,
    NB_BINS,
    GAMMA,
    ACCELERATOR,
    DEVICES,
    PRECISION,
    MAX_EPOCHS,
)
from loss import mse_loss

if __name__ == "__main__":
    dm = LegoDataModule(
        data_dir=DATA_DIR, img_size=(IMG_SIZE, IMG_SIZE), batch_size=1024, num_workers=3
    )
    model = NeRFLightning(LEARNING_RATE, TN, TF, NB_BINS, GAMMA)
    wandb_logger = WandbLogger(project="LegoNeRF", save_dir="logs/")

    trainer = L.Trainer(
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision=PRECISION,
        max_epochs=MAX_EPOCHS,
        min_epochs=0,
        fast_dev_run=False,
        reload_dataloaders_every_n_epochs=1,
        logger=wandb_logger,
    )

    trainer.fit(model=model, datamodule=dm)
