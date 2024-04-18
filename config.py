import torch


"""------------------------NeRF Config------------------------"""
# data
DATA_DIR: str = "dataset/lego/"
IMG_SIZE: int = 400
BATCH_SIZE: int = 3072

# model
LEARNING_RATE: float = 5e-4
TN: float = 2.0
TF: float = 6.0
NB_BINS: int = 192
GAMMA: float = 0.5

# trainer
ACCELERATOR: str = "gpu"
DEVICES: int = torch.cuda.device_count()
MAX_EPOCHS: int = 17
PRECISION: str = "32"
STRATEGY: str = "ddp"
LOGS_DIR: str = "logs/"

# 3d reconstruction
SCALE: float = 1.5

# eval
CKPT_DIR: str = "models/16_epoch_192_bins_400_nerf.ckpt"
CHUNK_SIZE: int = 20  # increase chunksize prevent CUDA out of memory errors
OUTPUTS_DIR: str = "tesings"


"""------------------------Sphere Config------------------------"""
# HEIGHT:int = 400
# WIDTH:int = 400
# FOCUS:int = 1200

# tn:float = 0.8
# tf:float = 1.2

ORIGIN: list = [0.0, 0.0, -1.0]
RADIUS: list = [0.1]
# learning_rate:float = 2e-1

color: list = [0.0, 1.0, 1.0]  # cyan
target_color: list = [1.0, 0.0, 1.0]  # purple
