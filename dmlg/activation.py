# activation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

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

    def get_device(self):
        return "cpu"
    
    def to_device(self):
        return self.to(self.get_device())
    
    def pretrain(self, epochs: int):
        device = self.get_device()
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        x_pre = torch.linspace(-10, 10, 4000).unsqueeze(1).to(device)

        for epoch in range(epochs):
            opt.zero_grad()
            y_true = F.silu(x_pre)
            y_pred = self(x_pre)
            loss = F.mse_loss(y_pred, y_true)
            loss.backward()
            opt.step()

            if epoch % 1000 == 0:
                print(f"Pretrain epoch {epoch}, loss = {loss.item():.6f}")
    
    def plot(self):
        device = self.get_device()

        with torch.no_grad():
            x_act = torch.linspace(-10, 10, 4000).unsqueeze(1).to(device)            
            y_default = F.silu(x_act).cpu()
            plt.figure(figsize=(10,5))            
            plt.plot(x_act.cpu(), y_default, label="SiLU")
    
            f_activation = self.to(device)
            y_activation = f_activation(x_act).cpu()
            plt.plot(x_act.cpu(), y_activation, label="Learned")

            plt.title("Activation Function")
            plt.legend()
            plt.show()        
        
class AMLPActivation(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x.unsqueeze(-1)).squeeze(-1)
