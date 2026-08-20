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

        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        h = F.silu(self.fc1(x))
        h = F.silu(self.fc2(h))
        h = F.silu(self.fc3(h))
        h = F.silu(self.fc4(h))
        return self.fc5(h)
