import torch
from torch import nn


class Nerf(nn.Module):
    def __init__(self, L_position = 10, L_direction = 4, hidden_dim=256):
        super(Nerf, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Linear(L_position * (2 * 3) + 3, hidden_dim), # 2 for sin and cos and 3 for all 3 dimensions of P in positional encoding formula
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
            nn.Linear(hidden_dim + (L_position * (2*3) + 3), hidden_dim), # skip connection iwth positional encodings from input added
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim + 1), # +1 for the sigma(density output) ps. doesn't depend on direction only the position
        )
        
        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_dim + L_direction * (2 * 3) + 3, hidden_dim // 2), # directional encodings added 
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3), 
            nn.Sigmoid()
        )
        
        self.L_position = L_position
        self.L_direction = L_direction
    
    def positional_encoding(self, x, L): #check positional encoding equation
        output = [x]
        for i in range(L):
            output.append(torch.sin(2 ** i * x))
            output.append(torch.cos(2 ** i * x))
            
        return torch.cat(output, dim=1)
    
    def forward(self, xyz, direction):
        x_embedding = self.positional_encoding(xyz, self.L_position) ## [batch_size, L_position * (2*3) + 3]
        direction_embedding = self.positional_encoding(direction, self.L_direction) # [batch_size, L_directio * (2*3) + 3]
        
        h = self.block1(x_embedding) # [batch, hidden_dim]
        h = self.block2(torch.cat((h, x_embedding), dim=1)) #[batch_Size, hidden_dim + 1]
        sigma = h[:, -1] #density
        h = h[:, :-1 ] #[batch_Size, hidden_dim]
        color = self.rgb_head(torch.cat((h, direction_embedding), dim=1))
        
        return color, torch.relu(sigma) #to get density always greater than zero
        
    def intersect(self, x, direction):
        return self.forward(x, direction)