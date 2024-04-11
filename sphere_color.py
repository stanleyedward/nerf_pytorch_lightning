import torch

from rendering import rendering
from utils import ray_generator
from model import Sphere
from config import (
    ORIGIN,
    RADIUS,
    HEIGHT,
    WIDTH,
    SAVE_DIR,
    FOCUS,
    target_color,
    color,
    learning_rate,
)
from sphere_train import sphere_train


def main():
    rays_origin, rays_direction = ray_generator(HEIGHT, WIDTH, FOCUS)

    target_sphere = Sphere(
        torch.tensor(ORIGIN), torch.tensor(RADIUS), torch.tensor(target_color)
    )
    target_px_colors = rendering(
        target_sphere,
        torch.tensor(rays_origin),
        torch.tensor(rays_direction),
        0.8,
        1.2,
        white_background=False,
    )

    color_to_optimize = torch.tensor(color, requires_grad=True, dtype=torch.float32)
    optimizer = torch.optim.SGD(params={color_to_optimize}, lr=learning_rate)

    loss = sphere_train(
        color_to_optimize,
        rays_origin,
        rays_direction,
        target_px_colors,
        optimizer,
        save_dir="test_img",
    )


if __name__ == "__main__":
    main()
