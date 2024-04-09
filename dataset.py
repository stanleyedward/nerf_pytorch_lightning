from torch.utils.data import Dataset
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
import numpy as np
from utils import get_ray_directions, get_rays
import os
import json
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from typing import Tuple
import lightning as L


class LegoDataModule(L.LightningDataModule):
    def __init__(self, data_dir, img_size: Tuple[int, int], batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None: ...

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.lego_train = LegoDataset(
                root_dir=self.data_dir, split="train", img_shape=self.img_size
            )
            self.lego_val = LegoDataset(
                root_dir=self.data_dir, split="val", img_shape=self.img_size
            )

        if stage == "test":
            self.lego_test = LegoDataset(
                root_dir=self.data_dir, split="test", img_shape=self.img_size
            )

        if stage == "predict":
            self.lego_pred = LegoDataset(
                root_dir=self.data_dir, split="val", img_shape=self.img_size
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.lego_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.lego_val, batch_size=1, num_workers=self.num_workers, shuffle=False
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.lego_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.lego_val, batch_size=1, num_workers=self.num_workers, shuffle=False
        )


class LegoDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        img_shape=(800, 800),
        white_background=True,
    ):
        assert img_shape[0] == img_shape[1], "image width must equal image height"

        self.tn = 2.0
        self.tf = 6.0

        self.root_dir = root_dir
        self.split = split
        self.img_shape = img_shape
        self.white_background = white_background

        self.transform = transforms.ToTensor()
        self.read_meta_data()

    def read_meta_data(self):
        with open(
            os.path.join(self.root_dir, f"transforms_{self.split}.json"), "r"
        ) as file:
            self.meta = json.load(file)

        focal = 0.5 * 800 / np.tan(0.5 * self.meta["camera_angle_x"])
        focal *= self.img_shape[0] / 800

        self.directions = get_ray_directions(
            self.img_shape[0], self.img_shape[1], focal
        )  # [H, W, 3]

        if self.split == "train":
            self.all_rays_origin = []
            self.all_rays_direction = []
            self.all_rgbs = []

            for frame in self.meta["frames"]:
                camera2world = torch.FloatTensor(frame["transform_matrix"])[:3, :4]

                img_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                img = Image.open(img_path)
                img = img.resize(self.img_shape, Image.LANCZOS)
                img = self.transform(img)  # [4, H, W]
                img = img.view(4, -1).permute(1, 0)  # [H*W, 4] RGBA
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # Blend A to RGB

                rays_origin, rays_direction = get_rays(
                    self.img_shape[0], self.img_shape[1], self.directions, camera2world
                )

                self.all_rays_origin.append(rays_origin)
                self.all_rays_direction.append(rays_direction)
                self.all_rgbs.append(img)

    def __len__(self):
        if self.split == "val":
            return 1  # valid single image
        return len(self.meta["frames"])

    def __getitem__(self, idx):
        if self.split == "train":
            # Use data in buffers
            sample = {
                "rays_origin": self.all_rays_origin[idx],
                "rays_direction": self.all_rays_direction[idx],
                "rgbs": self.all_rgbs[idx],
            }
        else:
            # Create data for each image separately
            frame = self.meta["frames"][idx]
            camera2world = torch.FloatTensor(frame["transform_matrix"])[:3, :4]

            img_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            img = Image.open(img_path)
            img = img.resize(self.img_shape, Image.LANCZOS)
            img = self.transform(img)  # [4, H, W]
            img = img.view(4, -1).permute(1, 0)  # [H*W, 4] RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # Blend A to RGB

            rays_origin, rays_direction = get_rays(
                self.img_shape[0], self.img_shape[1], self.directions, camera2world
            )

            sample = {
                "rays_origin": rays_origin,
                "rays_direction": rays_direction,
                "rgbs": img,
                "c2w": camera2world,
            }

        return sample
