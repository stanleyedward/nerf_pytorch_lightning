import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from typing import Tuple

def ray_generator(height:int, width:int, focus:int, camera2world=None) -> Tuple[np.ndarray, np.ndarray]:
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
    dirs: np.ndarray = np.stack((x,
                     -y, #-ve as we want y downwards
                     -z), axis=-1) # -ve as were looking down throught -z axis
    
    if camera2world is not None:
        dirs = (camera2world[:3, :3] @ dirs[..., None]).squeeze(-1)
        rays_origin += camera2world[:3, 3]
        
    #normalization
    rays_direction = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    rays_direction = rays_direction.reshape(-1, 3)
    
    return rays_origin, rays_direction


def plot_rays(origin, direction, t) -> None:
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    
    pt1 = origin
    pt2 = origin + t * direction 
    
    for p1, p2 in zip(pt1[::50], pt2[::50]): #plot 1/50
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])
    
    
    plt.xlabel('$X$ - coord')
    plt.ylabel('$Y$ - coord')
    
    plt.show()
    
def save_model(model:torch.nn.Module, target_dir:str, model_name:str) -> None:
    
    target_dir_path: Path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)
    
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pth' or '.pt'"
    model_save_path: Path = target_dir_path / model_name
    
    #save
    print(f"[INFO] saving model to : {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)
    