import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class PokerNet(nn.Module):
    def __init__(self, input_dim=2, output_dim=5):  # Now accepts input_dim
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.output = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.base(x)
        return self.output(x)

class CFRNetwork:
    def __init__(self, input_dim, device):  # Now accepts input_dim
        self.net = PokerNet(input_dim=input_dim).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        self.target_net = copy.deepcopy(self.net)
    
    def update_target(self, tau=0.001):
        """Soft update target network"""
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)