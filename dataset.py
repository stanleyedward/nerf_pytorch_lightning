from torch.utils.data import Dataset
import torch
import numpy as np
from utils import get_ray_directions, get_rays
import os
import json
from torchvision import transforms
from PIL import Image


class LegoDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        img_shape=(800, 800),
        white_background: bool = True,
    ):
        assert img_shape[0] == img_shape[1], "image width must equal image height"

        self.root_dir = root_dir
        self.split = split

        self.img_shape = img_shape
        self.white_background = white_background

        self.read_meta()
        self.transform = transforms.ToTensor()

    def read_meta_data(self):
        with open(
            os.path.join(self.root_dir, f"transforms_{self.split}.json"), "r"
        ) as file:
            self.meta = json.load(file)

        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta["camera_angle_x"])
        self.focal *= self.img_shape[0] / 800

        self.tn = 2.0
        self.tf = 6.0

        self.directions = get_ray_directions(
            self.img_shape[0], self.img_shape[1], self.focal
        )  # [H,W,3]

        if self.split == "train":
            self.all_rays = []
            self.all_rgbs = []
            self.valid_masks = []

            for frame in self.meta["frames"]:
                camera2world = torch.FloatTensor(frame["transform_matrix"])[:3, :4]

                img_path = os.join(self.root_dir, f"{frame['file_path)']}.png")
                img = Image.open(img_path)
                img = self.transform(img)  # [4, H, W]
                _valid_mask = (img[-1] > 0).flatten()  # [H*W] valid color area
                self.valid_masks += [_valid_mask]
                img = img.view(4, -1).permute(1, 0)  # rgba [H*W, 4]
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend a to rgbi
                self.all_rgbs += [img]

                rays_origin, rays_direction = get_rays(self.directions, camera2world)
                self.all_rays += [
                    torch.cat(
                        [
                            rays_origin,
                            rays_direction,
                            self.tn * torch.ones_like(rays_origin[:, :1]),
                            self.tf * torch.ones_like(rays_origin[:, :1]),
                        ],
                        1,
                    )
                ]

            self.all_rays = torch.cat(self.all_rays, 0)  # [len(meta['frames'])*H*W, 3]
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # [len(meta['frames'])*H*W, 3]
            self.valid_masks = torch.cat(
                self.valid_masks, 0
            )  # [len(meta['frames'])*H*W]

    def reduce_to_valid(self):
        self.all_rays = self.all_rays[self.valid_masks]
        self.all_rgbs = self.all_rgbs[self.valid_masks]

    def __len__(self):
        if self.split == "train":
            return len(self.all_rays)
        if self.split == "val":
            return 1  # valid single image

        return len(self.meta["frames"])

    def __getitem__(self, idx):
        if self.split == "train":  # use data in buffers
            sample = {"rays": self.all_rays[idx], "rgbs": self.all_rgbs[idx]}

        else:  # create data for each img sperately
            frame = self.meta["frames"][idx]
            camera2world = torch.FloatTensor(frame["transform_matrix"])[:3, :4]

            img_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            img = Image.open(img_path)
            img = img.resize(self.img_shape, Image.LANCZOS)
            img = self.transform(img)  # [4, H, w]
            valid_mask = (img[-1] > 0).flatten()  # [H*W] valid color region
            img = img.view(4, -1).permute(1, 0)  # [H*W, 4] RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB

            rays_origin, rays_direction = get_rays(self.directions, camera2world)

            rays = torch.cat(
                [
                    rays_origin,
                    rays_direction,
                    self.tn * torch.ones_like(rays_origin[:, :1]),
                    self.tf * torch.ones_like(rays_origin[:, :1]),
                ],
                1,
            )

            sample = {
                "rays": rays,
                "rgbs": img,
                "c2w": camera2world,
                "valid_mask": valid_mask,
            }

        return sample
