import torch 
from torch import nn
import matplotlib.pyplot as plt
from typing import List
import tqdm
import json

from rendering import rendering
from model import Sphere
from config import ORIGIN, RADIUS, HEIGHT, WIDTH

def sphere_train(color_to_optimize:torch.Tensor, rays_origin, rays_direction, target_px_colors, optimizer:torch.optim, save_outputs:bool = True) -> List:
    losses = []
    progress_bar = tqdm(
        iterable=range(200),
        total=200
    )

    for epoch in progress_bar:
        progress_bar.set_description(f"Epoch: {epoch}")
        s = Sphere(torch.tensor(ORIGIN), torch.tensor(RADIUS), color_to_optimize)
        Ax = rendering(s, torch.tensor(rays_origin),torch.tensor(rays_direction), 0.8, 1.2, white_background=False)
        
        loss = ((Ax - target_px_colors)**2).mean()
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        current_color = [round(color, 4) for color in color_to_optimize.data.tolist()]
        
        progress_bar.set_postfix(
            {
                "loss": loss.item(),
                "color": current_color
            }
        )
        
        if save_outputs:
            if epoch % 10 == 0:
                img = Ax.reshape(HEIGHT, WIDTH, 3).cpu().data.numpy()
                
                plt.title(f"{current_color}")
                plt.axis(False)
                plt.imshow(img)
                
                plt.savefig(f'sphere_img/sphere_epoch_{str(epoch)}.jpg', bbox_inches='tight')
                # plt.show()
                
    if save_outputs: 
        print(f"[INFO] figures saved to 'sphere_img/'")
        
    return losses


def sphere_train(color_to_optimize:torch.Tensor, rays_origin, rays_direction, target_px_colors, optimizer:torch.optim, save_outputs:bool = True) -> List:
    with open("sphere_losses.json", 'r') as file:
        t_losses = json.load(file)
    
    progress_bar = tqdm(
        iterable=range(200),
        total=200
    )

    for epoch in progress_bar:
        progress_bar.set_description(f"Epoch: {epoch}")
        s = Sphere(torch.tensor(ORIGIN), torch.tensor(RADIUS), color_to_optimize)
        Ax = rendering(s, torch.tensor(rays_origin),torch.tensor(rays_direction), 0.8, 1.2, white_background=False)
        
        loss = ((Ax - target_px_colors)**2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        current_color = [round(color, 4) for color in color_to_optimize.data.tolist()]
        
        progress_bar.set_postfix(
            {
                "loss": loss.item(),
                "color": current_color
            }
        )
        
        if epoch % 10 == 0:
            img = Ax.reshape(HEIGHT, WIDTH, 3).cpu().data.numpy()
            fig, axs = plt.subplots(1,2, figsize=(20,9))
            
            axs[0].imshow(img)
            axs[0].set_title(f"{current_color}")
            axs[0].axis(False)
            
            axs[1].plot(t_losses)
            axs[1].scatter(epoch, t_losses[epoch], c='m', label = 'epoch no.')
            axs[1].set(xlabel = 'epoch', ylabel='loss')
            axs[1].legend()
            
            plt.tight_layout()
            
            plt.savefig(f'sphere_img/sphere_epoch_{str(epoch)}.jpg', bbox_inches='tight')
        
        print(f"[INFO] figures saved to 'sphere_img/'")
