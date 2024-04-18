import torch
from tqdm import tqdm
from utils import test
from model import NeRFLightning
from dataset import LegoDataset
from config import (
    DATA_DIR,
    IMG_SIZE,
    TN,
    TF,
    NB_BINS,
    CKPT_DIR,
    CHUNK_SIZE,
    OUTPUTS_DIR,
)


device = "cuda" if torch.cuda.is_available() else "cpu"

lego_dataset = LegoDataset(
    root_dir=DATA_DIR, split="test", img_shape=(IMG_SIZE, IMG_SIZE)
)
lit_nerf = NeRFLightning.load_from_checkpoint(CKPT_DIR).to(device).eval()


if __name__ == "__main__":
    print(f"{'='*20}")
    print(f"Ouputs Directory: {OUTPUTS_DIR}/")
    print(f"{'='*20}\n")

    progress_bar = tqdm(range(0, 200), total=200)

    for idx in progress_bar:
        progress_bar.set_description(f"Image: {idx}")
        img, mse, psnr = test(
            lit_nerf,
            lego_dataset[idx]["rays_origin"]
            .reshape(-1, 3)
            .to(device)
            .type(torch.float),
            lego_dataset[idx]["rays_direction"]
            .reshape(-1, 3)
            .to(device)
            .type(torch.float),
            TN,
            TF,
            image_index=idx,
            nb_bins=NB_BINS,
            chunk_size=CHUNK_SIZE,
            height=lego_dataset.img_shape[0],
            width=lego_dataset.img_shape[1],
            target=lego_dataset[idx]["rgbs"].numpy(),
            outputs_dir=OUTPUTS_DIR,
            metrics=False,
        )
