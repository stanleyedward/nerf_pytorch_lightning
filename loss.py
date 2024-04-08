import numpy as np


def mse2psnr(mse_loss: float) -> float:
    return 20 * np.log10(1 / np.sqrt(mse_loss))


def mse_loss(image: np.ndarray, target_image: np.ndarray) -> float:
    return ((image - target_image) ** 2).mean()
