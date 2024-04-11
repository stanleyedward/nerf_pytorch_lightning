import torch
from torch import nn
import matplotlib.pyplot as plt
from typing import List
from tqdm import tqdm
import json
from pathlib import Path
import lightning as L
from dataset import LegoDataModule
from model import NeRFLightning

from rendering import rendering
from model import Sphere
from config import ORIGIN, RADIUS, HEIGHT, WIDTH, SAVE_DIR
from loss import mse_loss

if __name__ == '__main__':
    