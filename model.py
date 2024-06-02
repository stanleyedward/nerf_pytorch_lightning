import tinycudann as tcnn
import torch
from torch import nn
import lightning as L
from loss import mse2psnr, mse_loss
from rendering import rendering


class NeRFLightning(L.LightningModule):
    def __init__(self, learning_rate=1e-3, tn=2.0, tf=6.0, nb_bins=100, gamma=0.5):
        super().__init__()
        self.nerf = Nerf()
        self.learning_rate = learning_rate
        self.tn = tn
        self.tf = tf
        self.nb_bins = nb_bins
        self.gamma = gamma
        self.training_step_outputs = []

    def forward(self, xyz, direction):
        return self.nerf.forward(xyz, direction)

    def intersect(self, x, direction):
        return self.nerf.forward(x, direction)

    def _common_step(self, batch, batch_idx):
        rays_origin = batch[:, :3]
        rays_direction = batch[:, 3:6]
        target_img = batch[:, 6:]

        pred = rendering(
            self.nerf,
            rays_origin,
            rays_direction,
            self.tn,
            self.tf,
            self.nb_bins,
            device="cuda",
        )

        loss = mse_loss(pred, target_img)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.training_step_outputs.append(loss)
        self.log("loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        return loss

    def on_train_epoch_end(self) -> None:
        print("\n[INFO] Scheduler Step")
        self.scheduler.step()

        # avg_loss = (
        #     torch.stack([loss for loss in self.training_step_outputs]).mean().item()
        # )
        avg_psnr = (
            torch.stack([mse2psnr(loss) for loss in self.training_step_outputs])
            .mean()
            .item()
        )

        # self.log("loss", avg_loss)
        self.log("psnr", avg_psnr, prog_bar=True)

        self.training_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.nerf.parameters(), lr=self.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[2, 4, 8], gamma=self.gamma
        )
        return optimizer


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

class FullyFusedNerf(nn.Module):
    
    def __init__(self, Lpos=10, Ldir=4, hidden_dim=256):
        super(FullyFusedNerf, self).__init__()
        
        self.block1 = tcnn.Network(Lpos * 6 + 3, 
                                   hidden_dim, 
                                   {"otype": "FullyFusedMLP",
                                    "activation": "ReLU",
                                    "output_activation": "ReLU",
                                    "n_neurons": hidden_dim,
                                    "n_hidden_layers": 4},)
        
        self.block2 = tcnn.Network(hidden_dim + Lpos * 6 + 3, 
                                   hidden_dim + 1, 
                                   {"otype": "FullyFusedMLP",
                                    "activation": "ReLU",
                                    "output_activation": "None",
                                    "n_neurons": hidden_dim,
                                    "n_hidden_layers": 3},)
        
        
        self.rgb_head = tcnn.Network(hidden_dim + Ldir * 6 + 3, 
                                   3, 
                                   {"otype": "FullyFusedMLP",
                                    "activation": "ReLU",
                                    "output_activation": "Sigmoid",
                                    "n_neurons": hidden_dim // 2,
                                    "n_hidden_layers": 1},)
        
        self.Lpos = Lpos
        self.Ldir = Ldir
        
    def positional_encoding(self, x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)
            
                                    
        
    def forward(self, xyz, d):
        
        x_emb = self.positional_encoding(xyz, self.Lpos) # [batch_size, Lpos * 6 + 3]
        d_emb = self.positional_encoding(d, self.Ldir) # [batch_size, Ldir * 6 + 3]
        
        h = self.block1(x_emb) # [batch_size, hidden_dim]
        h = self.block2(torch.cat((h, x_emb), dim=1)) # [batch_size, hidden_dim + 1]
        sigma = h[:, -1]
        h = h[:, :-1] # [batch_size, hidden_dim]
        c = self.rgb_head(torch.cat((h, d_emb), dim=1))
        
        return c, torch.relu(sigma)
        
    
    def intersect(self, x, d):
        return self.forward(x, d)
