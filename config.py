HEIGHT = 400
WIDTH = 400
FOCUS = 1200

tn = 0.8
tf = 1.2

ORIGIN = [0.0, 0.0, -1.0]
RADIUS = [0.1]
learning_rate = 2e-1

color = [0.0, 1.0, 1.0]  # cyan
target_color = [1.0, 0.0, 1.0]  # purple

SAVE_DIR = "images"

#data
DATA_DIR = 'dataset/lego/'
IMG_SIZE = 400
BATCH_SIZE = 1024

#model
LEARNING_RATE = 1e-3
TN = 2
TF = 6
NB_BINS = 100
GAMMA = 0.5

#trainer
ACCELERATOR = 'gpu'
DEVICES = [0]  # int(torch.cuda.isavailable())
MAX_EPOCHS = 16
PRECISION = "32"
