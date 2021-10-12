import torch
import torch.nn as nn


class NET(nn.Module):
    """定义神经网络类"""
    def __init__(self, state_dim, action_dim):
        super(NET, self).__init__()
        self.lin1 = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU()
        )
        self.lin2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU()
        )
        self.lin3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.lin_a = nn.Linear(256, action_dim)
        self.lin_v = nn.Linear(256, 1)

    def forward(self, state):
        state = state.to(torch.float32)
        feature = self.lin1(state)
        feature = self.lin2(feature)
        feature = self.lin3(feature)
        V = self.lin_v(feature)
        A = self.lin_a(feature)
        Q = V + (A - torch.mean(A, dim=1, keepdim=True))
        return Q

    def advantage(self, state):
        state = state.to(torch.float32)
        feature = self.lin1(state)
        feature = self.lin2(feature)
        feature = self.lin3(feature)
        A = self.lin_a(feature)
        return A
