import torch


def rendering(
    model,
    rays_origin: torch.Tensor,
    rays_direction: torch.Tensor,
    tn: float,
    tf: float,
    nb_bins: int = 100,
    device: str = "cpu",
    white_background: bool = True,
) -> torch.Tensor:
    t = torch.linspace(tn, tf, nb_bins).to(device)  # [nb_bins]
    delta = torch.cat(
        (t[1:] - t[:-1], torch.tensor([1e10], device=device))
    )  # size is no of bins - 1 so we concat infinity as the last value, most nerf papers take the last delta value as infinity
    x = rays_origin.unsqueeze(1) + t.unsqueeze(0).unsqueeze(
        -1
    ) * rays_direction.unsqueeze(1)

    if type(model).__name__ == "Nerf" or type(model).__name__ == "NeRFLightning":
        colors, density = model.intersect(
            x.reshape(-1, 3),
            rays_direction.expand(x.shape[1], x.shape[0], 3)
            .transpose(0, 1)
            .reshape(-1, 3),
        )

    elif type(model).__name__ == "Voxels" or type(model).__name__ == "Sphere":
        colors, density = model.intersect(x.reshape(-1, 3))

    else:
        raise Exception("Model not found.")

    colors = colors.reshape((x.shape[0], nb_bins, 3))  # [nb_rays, nb_bins, 3]
    density = density.reshape((x.shape[0], nb_bins, 1))  # [nb_rays, nb_bins, 1]

    alpha = 1 - torch.exp(
        -density.squeeze() * delta.unsqueeze(0)
    )  # shape [nb_rays, nb_bins, 1]
    T = compute_accumulated_transmittance(1 - alpha)  # [nb_rays, nb_bins, 1]
    weights = T * alpha  # [nb_rays, nb_bins]

    if white_background:
        color = (weights.unsqueeze(-1) * colors).sum(1)  # [nb_rays, 3]
        weight_sum = weights.sum(
            -1
        )  # [nb_rays] tells if we are in empty space or no via accumulation of denity # when using white background regularization

        return color + 1 - weight_sum.unsqueeze(-1)

    else:
        color = (weights.unsqueeze(-1) * colors).sum(1)  # shape [nb_rays, 3]

    return color


def compute_accumulated_transmittance(betas: torch.Tensor) -> torch.Tensor:
    accumulated_transmittance = torch.cumprod(betas, 1)
    return torch.cat(
        (
            torch.ones(
                accumulated_transmittance.shape[0],
                1,
                device=accumulated_transmittance.device,
            ),  # since we shift to the right
            accumulated_transmittance[:, :-1],
        ),
        dim=1,
    )  # sum goes from i =1 to i= N-1
