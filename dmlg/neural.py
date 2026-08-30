# neural.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Main MLP
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, first_hidden_size, other_hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.first_hidden_size = first_hidden_size
        self.other_hidden_size = other_hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, first_hidden_size)
        self.fc2 = nn.Linear(first_hidden_size, other_hidden_size)
        self.fc3 = nn.Linear(other_hidden_size, other_hidden_size)
        self.fc4 = nn.Linear(other_hidden_size, other_hidden_size)
        self.fc5 = nn.Linear(other_hidden_size, output_size)

        self.fc_gate = nn.Linear(other_hidden_size, other_hidden_size)

        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)

    # r-skip + r-gate hybrid forward
    def forward(self, x):
        # initial projection
        h = F.silu(self.fc1(x))
        h = F.silu(self.fc2(h))

        # R-gate + R-skip block
        for _ in range(3):
            # gate ∈ (0,1)
            g = torch.sigmoid(self.fc_gate(h))
            # recursive update
            u = F.silu(self.fc3(h))
            # gated residual: h ← h + g·(u−h)
            h = h + g * (u - h)

        # extra residual refinement
        h4 = F.silu(self.fc4(h))
        h = h + h4

        return self.fc5(h)
