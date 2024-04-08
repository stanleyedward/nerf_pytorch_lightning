import torch
from torch import nn


class Nerf(nn.Module):
    def __init__(self, L_position=10, L_direction=4, hidden_dim=256):
        super(Nerf, self).__init__()

        self.block1 = nn.Sequential(
            nn.Linear(
                L_position * (2 * 3) + 3, hidden_dim
            ),  # 2 for sin and cos and 3 for all 3 dimensions of P in positional encoding formula
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Linear(
                hidden_dim + (L_position * (2 * 3) + 3), hidden_dim
            ),  # skip connection iwth positional encodings from input added
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(
                hidden_dim, hidden_dim + 1
            ),  # +1 for the sigma(density output) ps. doesn't depend on direction only the position
        )

        self.rgb_head = nn.Sequential(
            nn.Linear(
                hidden_dim + L_direction * (2 * 3) + 3, hidden_dim // 2
            ),  # directional encodings added
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )

        self.L_position = L_position
        self.L_direction = L_direction

    def positional_encoding(self, x, L):  # check positional encoding equation
        output = [x]
        for i in range(L):
            output.append(torch.sin(2**i * x))
            output.append(torch.cos(2**i * x))

        return torch.cat(output, dim=1)

    def forward(self, xyz, direction):
        x_embedding = self.positional_encoding(
            xyz, self.L_position
        )  ## [batch_size, L_position * (2*3) + 3]
        direction_embedding = self.positional_encoding(
            direction, self.L_direction
        )  # [batch_size, L_directio * (2*3) + 3]

        h = self.block1(x_embedding)  # [batch, hidden_dim]
        h = self.block2(
            torch.cat((h, x_embedding), dim=1)
        )  # [batch_Size, hidden_dim + 1]
        sigma = h[:, -1]  # density
        h = h[:, :-1]  # [batch_Size, hidden_dim]
        color = self.rgb_head(torch.cat((h, direction_embedding), dim=1))

        return color, torch.relu(sigma)  # to get density always greater than zero

    def intersect(self, x, direction):
        return self.forward(x, direction)


class Sphere:
    def __init__(self, position, radius, color):
        self.position = position
        self.radius = radius
        self.color = color

    def intersect(self, x):  # tells if ray hits the sphere or not
        """_summary_

        Args:
            x (parameter): points shape [batch_size, 3]

        Returns:
            _type_: _description_
        """
        # check if inside the sphere
        # (x-xo)^2 + (y-yo)^2 + (z-zo)^2 < r^2
        condition = (x[:, 0] - self.position[0]) ** 2 + (
            x[:, 1] - self.position[1]
        ) ** 2 + (x[:, 2] - self.position[2]) ** 2 <= self.radius**2

        number_of_rays = x.shape[0]
        colors = torch.zeros((number_of_rays, 3))
        density = torch.zeros((number_of_rays, 1))

        colors[condition] = self.color  # assign color where condition passes
        density[condition] = 10  # assume constant

        return colors, density


class Voxels(nn.Module):
    def __init__(self, nb_voxels: int = 100, scale=1, device="cpu"):
        super(Voxels, self).__init__()

        self.voxels = torch.nn.Parameter(
            torch.rand(
                (nb_voxels, nb_voxels, nb_voxels, 4), device=device, requires_grad=True
            )
        )  # colors and density cannot be negative therefore dont use normal distribtion

        self.nb_voxels = nb_voxels
        self.device = device
        self.scale = scale

    def forward(self, xyz):
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        condition = (
            (x.abs() < (self.scale / 2))
            & (y.abs() < (self.scale / 2))
            & (z.abs() < (self.scale / 2))
        )

        colors_and_densities = torch.zeros((xyz.shape[0], 4), device=xyz.device)

        idx_x = (
            x[condition] / (self.scale / self.nb_voxels) + self.nb_voxels / 2
        ).type(torch.long)
        idx_y = (
            y[condition] / (self.scale / self.nb_voxels) + self.nb_voxels / 2
        ).type(torch.long)
        idx_z = (
            z[condition] / (self.scale / self.nb_voxels) + self.nb_voxels / 2
        ).type(torch.long)

        colors_and_densities[condition, :3] = self.voxels[idx_x, idx_y, idx_z, :3]
        colors_and_densities[condition, -1] = self.voxels[idx_x, idx_y, idx_z, -1]

        return torch.sigmoid(colors_and_densities[:, :3]), torch.relu(
            colors_and_densities[:, -1:]
        )

    def intersect(self, x):
        return self.forward(x)
