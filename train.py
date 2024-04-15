import lightning as L
from lightning.pytorch.loggers import WandbLogger

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
)


if __name__ == "__main__":
    dm = LegoDataModule(
        data_dir=DATA_DIR, img_size=(400, 400), batch_size=1024, num_workers=3
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
        logger=None,
    )

    trainer.fit(model=model, datamodule=dm)
