import torch 
from torch import nn
import matplotlib.pyplot as plt
from typing import List
import tqdm

from rendering import rendering
from model import Sphere
from config import ORIGIN, RADIUS, HEIGHT, WIDTH

def sphere_train(color_to_optimize:torch.Tensor, rays_origin, rays_direction, target_px_colors, optimizer:torch.optim, save_outputs:bool = True) -> List:
    losses = []
    for epoch in range(200):
        
        s = Sphere(torch.tensor(ORIGIN), torch.tensor(RADIUS), color_to_optimize)
        Ax = rendering(s, torch.tensor(rays_origin),torch.tensor(rays_direction), 0.8, 1.2, white_background=False)

        loss = ((Ax - target_px_colors)**2).mean()
        losses.append(loss.item())
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if save_outputs:
            if epoch % 10 == 0:
                current_color = [round(color, 4) for color in color_to_optimize.data.tolist()]
                print(f"loss: {loss:.4f}")
                print(f"color: {current_color}")
                
                
                img = Ax.reshape(HEIGHT, WIDTH, 3).cpu().data.numpy()
                plt.title(f"{current_color}")
                plt.axis(False)
                plt.imshow(img)
                # plt.show()
                plt.savefig(f'sphere_img/sphere_epoch_{str(epoch)}.jpg', bbox_inches='tight')
            
    return losses