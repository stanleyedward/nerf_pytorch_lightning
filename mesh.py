import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import pyglet

import dataset
import rendering
import model
import train

import mcubes
import trimesh
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
    SCALE,
)

if __name__ == '__main__':
    trimesh.load_mesh("models/nerf_mesh.obj").show()