from typing import Union
import torch

def mse2psnr(mse_loss: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
    return 20 * torch.log10(1 / torch.sqrt(mse_loss.cpu().detach()))

def mse_loss(image: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
    return ((image - target_image) ** 2).mean()

# import numpy as np


# def mse2psnr(mse_loss):
#     return 20 * np.log10(1 / np.sqrt(mse_loss.cpu().detach()))


# def mse_loss(image, target_image):
#     return ((image - target_image) ** 2).mean()

