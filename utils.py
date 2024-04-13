import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from typing import Tuple
from pathlib import Path
from rendering import rendering
from loss import mse2psnr


def plot_rays(origin, direction, t) -> None:
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")

    pt1 = origin
    pt2 = origin + t * direction

    for p1, p2 in zip(pt1[::50], pt2[::50]):  # plot 1/50
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])

    plt.xlabel("$X$ - coord")
    plt.ylabel("$Y$ - coord")

    plt.show()


def save_model(model: torch.nn.Module, target_dir: str, model_name: str) -> None:

    target_dir_path: Path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pth' or '.pt'"
    model_save_path: Path = target_dir_path / model_name

    # save
    print(f"[INFO] saving model to : {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def get_ray_directions(height: int, width: int, focus: int):
    # coord grid
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)  # u.shape & v.shape = [H, W]

    # direction
    x = u - (width / 2)
    y = v - (height / 2)
    z = np.ones_like(u) * focus

    directions = np.stack(
        (x, -y, -z), axis=-1  # -ve as we want y downwards
    )  # -ve as were looking down throught -z axis

    return directions


def get_rays(height: int, width: int, directions, camera2world=None):
    rays_origin = np.zeros((height * width, 3))
    rays_direction = np.zeros((height * width, 3))

    if camera2world is not None:
        directions = (camera2world[:3, :3] @ directions[..., None]).squeeze(-1)
        rays_origin += camera2world[:3, 3].numpy()

    # normalization
    rays_direction = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    rays_direction = rays_direction.reshape(-1, 3)

    return torch.tensor(rays_origin), torch.tensor(rays_direction)


def set_seeds(seed: int = 42):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


@torch.inference_mode()
def test(
    model,
    origin,
    direction,
    tn,
    tf,
    image_index,
    nb_bins=100,
    chunk_size=20,
    height=400,
    width=400,
    target=None,
    outputs_dir=None,
    title=True,
):
    """_summary_

    Args:
        model (_type_): _description_
        origin (_type_): [Height*Width, 3]
        direction (_type_): [Height*Width, 3]
        tn (_type_): _description_
        tf (_type_): _description_
        image_index (_type_): _description_
        nb_bins (int, optional): _description_. Defaults to 100.
        chunk_size (int, optional): _description_. Defaults to 20.
        height (int, optional): _description_. Defaults to 400.
        width (int, optional): _description_. Defaults to 400.
        target (_type_, optional): [Height,Width, 3]. Defaults to None.
        outputs_dir (_type_, optional): _description_. Defaults to None.
        title (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    save_dir = Path(outputs_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    with torch.inference_mode():
        model.eval()
        origin = origin.chunk(chunk_size)
        direction = direction.chunk(chunk_size)

        image = []
        for origin_batch, direction_batch in zip(
            origin, direction
        ):  # zip to interate both lists at the same time
            img_batch = rendering(
                model,
                origin_batch,
                direction_batch,
                tn,
                tf,
                nb_bins,
                device=origin_batch.device,
            )
            image.append(img_batch)  # [N, 3]

        image = torch.cat(image)  # [H*W, 3]
        image = (
            image.reshape(height, width, 3).cpu().numpy()
        )  # no need for .data as there are no gradients!

        if target is not None:
            loss = ((image - target) ** 2).mean()  # same as mean squared errorfunction
            psnr = mse2psnr(loss)

            if outputs_dir is not None:
                if title:
                    plt.title(f"MSE: {loss:.4f} || PSNR: {psnr:.4f}")
                plt.imshow(image)
                plt.savefig(
                    f"{outputs_dir}/lego_{image_index}.png", bbox_inches="tight"
                )
            return image, loss, psnr

        else:
            if outputs_dir is not None:
                plt.imshow(image)
                plt.savefig(
                    f"{outputs_dir}/lego_{image_index}.png", bbox_inches="tight"
                )
            return image
