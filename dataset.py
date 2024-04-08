from torch.utils.data import Dataset
import torch
import numpy as np
from utils import ray_generator
import os
import json
from torchvision import transforms
from PIL import Image

class LegoDataset(Dataset):
    def __init__(self, root_dir, split='train', img_shape=(800, 800), white_background:bool = True):
        assert img_shape[0] == img_shape[1] , 'image width must equal image height'
        
        self.root_dir = root_dir
        self.split = split
        
        self.img_shape = img_shape
        self.white_background = white_background
        
        self.read_meta()
        self.transform = transforms.ToTensor()
        
    def read_meta_data(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split}.json"), 'r') as file:
            self.meta = json.load(file)
        
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x'])
        self.focal *= self.img_shape[0]/800
        
        self.tn = 2.0
        self.tf = 6.0
        
        self.direction = get_ray_directions(self.img_shape[0], self.img_shape[1], self.focal) # [H,W,3]
        
        if self.split == 'train':
            self.all_rays = []
            self.all_rgbs = []
            self.valid_masks = []
            
            for frame in self.meta['frames']:
                camera2world = torch.FloatTensor(frame['transform_matrix'])[:3, :4]
                
                img_path = os.join(self.root_dir, f"{frame['file_path)']}.png")
                img = Image.open(img_path)
                img = self.transform(img) #[4, H, W]
                _valid_mask = (img[-1] > 0).flatten() # [H*W] valid color area
                self.valid_masks += [_valid_mask]
                img = img.view(4, -1).permute(1, 0) # rgba [H*W, 4] 
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) #blend a to rgbi
                self.all_rgbs += [img]
                
                rays_origin, rays_direction = get_rays(self.direction, camera2world)
                self.all_rays += [torch.cat([rays_origin, rays_direction,
                                             self.tn*torch.ones_like(rays_origin[:, :1]),
                                             self.tf*torch.ones_like(rays_origin[:, :1])],
                                            1)]
            
            self.all_rays = torch.cat(self.all_rays, 0) #[len(meta['frames'])*H*W, 3]
            self.all_rgbs = torch.cat(self.all_rgbs, 0) #[len(meta['frames'])*H*W, 3]
            self.valid_masks = torch.cat(self.valid_masks, 0) #[len(meta['frames'])*H*W]
            
    def reduce_to_valid(self):
        self.all_rays = self.all_rays[self.valid_masks]
        self.all_rgbs = self.all_rgbs[self.valid_masks]
            
    def __len__():
        ...
    
    def __getitem__(self, idx):
        ...
    
    

def get_ray_directions(height:int, width:int, focus:int) -> np.ndarray:
    rays_origin: np.ndarray = np.zeros((height * width, 3))
    rays_direction: np.ndarray = np.zeros((height * width, 3)) 
    
    #coord grid
    u: np.ndarray = np.arange(width)
    v: np.ndarray = np.arange(height)
    u,v = np.meshgrid(u, v) # u.shape & v.shape = [H, W]
    
    #direction
    x: np.ndarray = u - (width/2)
    y: np.ndarray = v - (height/2)
    z: np.ndarray = np.ones_like(u) * focus
    directions: np.ndarray = np.stack((x,
                     -y, #-ve as we want y downwards
                     -z), axis=-1) # -ve as were looking down throught -z axis
    
    return directions

def get_rays(height:int, width:int, directions:np.ndarray, camera2world:np.ndarray = None):
    rays_origin: np.ndarray = np.zeros((height * width, 3))
    rays_direction: np.ndarray = np.zeros((height * width, 3)) 
    
    if camera2world is not None:
        directions = (camera2world[:3, :3] @ directions[..., None]).squeeze(-1)
        rays_origin += camera2world[:3, 3]
        
    #normalization
    rays_direction = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    rays_direction = rays_direction.reshape(-1, 3)
    
    return rays_origin, rays_direction
