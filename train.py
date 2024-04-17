import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb

from dataset import LegoDataModule
from model import NeRFLightning
from config import (
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
    BATCH_SIZE,
    LOGS_DIR,
)


if __name__ == "__main__":
    dm = LegoDataModule(
        data_dir=DATA_DIR,
        img_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        num_workers=3,
    )
    model = NeRFLightning(LEARNING_RATE, TN, TF, NB_BINS, GAMMA)
    wandb_logger = WandbLogger(project="LegoNeRF", save_dir=LOGS_DIR)

    trainer = L.Trainer(
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision=PRECISION,
        max_epochs=MAX_EPOCHS,
        min_epochs=0,
        fast_dev_run=False,
        reload_dataloaders_every_n_epochs=1,
        logger=False,
    )

    trainer.fit(model=model, datamodule=dm)
wandb.finish()
