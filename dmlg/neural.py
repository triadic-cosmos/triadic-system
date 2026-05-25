# neural.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActivationMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        return self.fc3(x)

class AMLPActivation(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x.unsqueeze(-1)).squeeze(-1)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, activation_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_size = activation_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        if activation_size > 0:
            self.act_mlp = ActivationMLP(activation_size)
            self.act = AMLPActivation(self.act_mlp)
        else:
            self.act = F.silu # use fixed activation function

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        return self.fc3(h)

    def add_output_channel(self):
        new = NeuralNetwork(self.input_size, self.hidden_size, self.activation_size, self.output_size + 1)
        with torch.no_grad():
            new.fc1.weight.copy_(self.fc1.weight)
            new.fc1.bias.copy_(self.fc1.bias)
            new.fc2.weight.copy_(self.fc2.weight)
            new.fc2.bias.copy_(self.fc2.bias)
            new.fc3.weight[:-1].copy_(self.fc3.weight)
            new.fc3.bias[:-1].copy_(self.fc3.bias)
            new.fc3.weight[-1].zero_()
            new.fc3.bias[-1].zero_()
        return new
