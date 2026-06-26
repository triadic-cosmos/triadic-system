# neural.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Activation MLP
class ActivationMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return self.fc3(x)

class AMLPActivation(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x.unsqueeze(-1)).squeeze(-1)

# Main MLP
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        self.activation: ActivationMLP = activation
        self.act = AMLPActivation(activation)

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        return self.fc3(h)
