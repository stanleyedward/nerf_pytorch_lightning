import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

def ray_generator(height, width, focus):
    rays_origin = np.zeros((height * width, 3))
    rays_direction = np.zeros((height * width, 3)) 
    
    #coord grid
    u = np.arange(width)
    v = np.arange(height)
    u,v = np.meshgrid(u, v) # u.shape & v.shape = [H, W]
    
    #direction
    x = u - (width/2)
    y = v - (height/2)
    z = np.ones_like(u) * focus
    dirs = np.stack((x,
                     -y, #-ve as we want y downwards
                     -z), axis=-1) # -ve as were looking down throught -z axis
    
    #normalization
    rays_direction = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    rays_direction = rays_direction.reshape(-1, 3)
    
    return rays_origin, rays_direction


def plot_rays(origin, direction, t):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    
    pt1 = origin
    pt2 = origin + t * direction 
    
    for p1, p2 in zip(pt1[::50], pt2[::50]): #plot 1/50
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])
    
    
    plt.xlabel('$X$ - coord')
    plt.ylabel('$Y$ - coord')
    
    plt.show()
    
def save_model(model, target_dir, model_name:str):
    
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)
    
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pth' or '.pt'"
    model_save_path = target_dir_path / model_name
    
    #save
    print(f"[INFO] saving model to : {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)
    