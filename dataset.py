from torch.utils.data import Dataset
from utils import ray_generator
import os
import json

class LegoDataset(Dataset):
    def __init__(self, root_dir, split='train', img_shape=(800, 800), white_background:bool = True):
        assert img_shape[0] == img_shape[1] , 'image width must equal image height'
        
        self.root_dir = root_dir
        self.split = split
        
        self.img_shape = img_shape
        self.white_background = white_background
        
        self.define_transforms()
        self.read_meta()
        
    def read_meta_data(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split}.json"), 'r') as file:
            self.meta = json.load(file)
            
    def __len__():
        ...
    
    def __getitem__(self, idx):
        ...