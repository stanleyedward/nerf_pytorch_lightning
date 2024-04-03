import numpy as np

def mse2psnr(mse_loss):
    return 20 * np.log10(1 / np.sqrt(mse_loss))

def mse_loss(image, target_image):
    loss = ((image - target_image)**2).mean()
    return loss