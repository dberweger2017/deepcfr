import torch
import torch.nn as nn

class BettingHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.discrete_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)  # 6 discrete bet sizes
        )
        self.continuous_head = nn.Sequential(
            nn.Linear(input_dim + 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        discrete_logits = self.discrete_head(x)
        continuous_adj = self.continuous_head(
            torch.cat([x, discrete_logits], dim=-1))
        return discrete_logits, continuous_adj

class NLHEPolicyNetwork(nn.Module):
    def __init__(self, state_dim=28):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
        )
        self.action_head = nn.Linear(128, 3)  # FOLD/CALL/RAISE
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.base(x)
        return self.action_head(x), self.value_head(x)