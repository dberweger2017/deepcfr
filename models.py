import torch
import torch.nn as nn
import torch.nn.functional as F

class PokerNet(nn.Module):
    def __init__(self, input_dim=54, output_dim=5):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        self.policy_head = nn.Linear(256, output_dim)
        self.value_head = nn.Linear(256, 1)
        
    def forward(self, x):
        x = self.base(x)
        return self.policy_head(x), self.value_head(x)

class CFRNetwork:
    def __init__(self, device):
        self.net = PokerNet().to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)