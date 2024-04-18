import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Union
from loss import mse2psnr
from rendering import rendering


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


def get_ray_directions(height: int, width: int, focus: int) -> np.ndarray:
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


def get_rays(
    height: int, width: int, directions: np.ndarray, camera2world=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute rays origin and direction.

    Args:
        height (int): Height of the image.
        width (int): Width of the image.
        directions (np.ndarray): Array of shape (height, width, 3) representing the ray directions.
        camera2world (Optional[np.ndarray], optional): Camera to world transformation matrix. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing the rays origin and direction as torch tensors.
    """
    rays_origin = np.zeros((height * width, 3))
    rays_direction = np.zeros((height * width, 3))

    if camera2world is not None:
        directions = (camera2world[:3, :3] @ directions[..., None]).squeeze(-1)
        rays_origin += camera2world[:3, 3].numpy()

    # normalization
    rays_direction = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    rays_direction = rays_direction.reshape(-1, 3)

    return (
        torch.Tensor(rays_origin).clone().detach(),
        torch.Tensor(rays_direction).clone().detach(),
    )


def set_seeds(seed: int = 42) -> None:

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


@torch.inference_mode()
def test(
    model,
    origin: torch.Tensor,
    direction: torch.Tensor,
    tn: float,
    tf: float,
    image_index: int,
    nb_bins: int = 100,
    chunk_size: int = 20,
    height: int = 400,
    width: int = 400,
    target: Optional[torch.Tensor] = None,
    outputs_dir: Optional[str] = None,
    metrics: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, float, float]]:
    """_summary_

    Args:
        model (torch.nn.Module): _description_
        origin (torch.Tensor): [Height*Width, 3]
        direction (torch.Tensor): [Height*Width, 3]
        tn (float): _description_
        tf (float): _description_
        image_index (int): _description_
        nb_bins (int, optional): _description_. Defaults to 100.
        chunk_size (int, optional): _description_. Defaults to 20.
        height (int, optional): _description_. Defaults to 400.
        width (int, optional): _description_. Defaults to 400.
        target (Optional[torch.Tensor], optional): [Height,Width, 3]. Defaults to None.
        outputs_dir (Optional[str], optional): _description_. Defaults to None.
        metrics (bool, optional): _description_. Defaults to True.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, float, float]]: _description_
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
            psnr = mse2psnr(torch.tensor(loss))
            if outputs_dir is not None:
                if metrics:
                    plt.title(f"MSE: {loss:.4f} || PSNR: {psnr:.4f}")
                plt.axis(False)
                plt.imshow(image.clip(0, 1))
                plt.savefig(
                    f"{outputs_dir}/lego_{image_index}.png", bbox_inches="tight"
                )
            return image, loss, psnr
        else:
            if outputs_dir is not None:
                plt.imshow(image.clip(0, 1))
                plt.savefig(
                    f"{outputs_dir}/lego_{image_index}.png", bbox_inches="tight"
                )
            return image


def training(
    model, optimizer, scheduler, dataloader, tn, tf, nb_bins, nb_epochs, device="cpu"
):

    training_loss = []

    progress_bar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
    )

    for epoch in range(nb_epochs):
        progress_bar.set_description(f"Training Epoch: {epoch}")
        for idx, batch in progress_bar:
            origin = batch[:, :3].to(device)
            direction = batch[:, 3:6].to(device)

            target = batch[:, 6:].to(device)

            prediction = rendering(model, origin, direction, tn, tf, nb_bins, device)

            loss = ((prediction - target) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({"loss": loss.item()})
            training_loss.append(loss.item())

        scheduler.step()

        torch.save(model.cpu(), "models/model_nerf")
        model.to(device)


def get_state_dict(
    model: torch.nn.Module, ckpt_dir: str, save_path=Union[None, str]
) -> torch.nn.Module:
    checkpoint = torch.load(ckpt_dir)
    # since keys dont align properly here
    mapped_state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("nerf."):  # If checkpoint keys start with "nerf.", remove it
            mapped_state_dict[k[len("nerf.") :]] = v
        else:
            mapped_state_dict[k] = v

    model.load_state_dict(mapped_state_dict)

    if save_path is not None:
        torch.save(obj=model.state_dict(), f=save_path)

    return model


def get_avg_metrics(outputs_json_dir: str) -> Tuple[float, float]:
    with open("outputs/nerf_testing.json", "r") as f:
        data = json.load(f)

    avg_psnr = sum(data["psnr"]) / len(data["psnr"])
    avg_mse = sum(data["mse"]) / len(data["mse"])
    return avg_psnr, avg_mse
