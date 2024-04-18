from torch.utils.data import Dataset
import torch
import numpy as np
from utils import get_ray_directions, get_rays
import os
import json
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import lightning as L
from rendering import rendering
from utils import test
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
    CKPT_DIR,
    CHUNK_SIZE,
    OUTPUTS_DIR
)

from dataset import LegoDataset, LegoDataModule
from model import Nerf

device = "cuda" if torch.cuda.is_available() else "cpu"
lego_dataset = LegoDataset(root_dir="dataset/lego/", split="test", img_shape=(IMG_SIZE, IMG_SIZE))
lit_nerf = NeRFLightning.load_from_checkpoint(CKPT_DIR).to(device).eval()


if __name__ == '__main__':
    print(f"{'='*5}Outputs in {OUTPUTS_DIR}{'='*5}")
    progress_bar = tqdm(range(0, 200),
                        total=200)
    for idx in progress_bar:
        progress_bar.set_description(f"Image: {idx}")
        img, mse, psnr = test(
            lit_nerf,
            lego_dataset[idx]["rays_origin"].reshape(-1, 3).to(device).type(torch.float),
            lego_dataset[idx]["rays_direction"].reshape(-1, 3).to(device).type(torch.float),
            TN,
            TF,
            image_index=idx,
            nb_bins=NB_BINS,
            chunk_size=CHUNK_SIZE,
            height=lego_dataset.img_shape[0],
            width=lego_dataset.img_shape[1],
            target=lego_dataset[idx]["rgbs"].numpy(),
            outputs_dir=OUTPUTS_DIR,
            metrics=False
        )    
