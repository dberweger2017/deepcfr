import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class PokerNet(nn.Module):
    def __init__(self, input_dim=54, output_dim=5):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(input_dim=512, out_features=256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        self.policy_head = nn.Linear(256, output_dim)
        # Note: The value_head was removed as recommended.

    def forward(self, x):
        x = self.base(x)
        return self.policy_head(x)

class CFRNetwork:
    def __init__(self, device):
        self.net = PokerNet().to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        self.target_net = copy.deepcopy(self.net)  # For rolling opponent updates

    def update_target(self, tau=0.001):
        # Soft update: target_net = tau * net + (1 - tau) * target_net
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)