# Demonstration of adaptive activation functions using a trainable MLP

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ============================================================
# Settings
# ============================================================

neuron_size: int = 8
model_size: int = 16
activation_epochs: int = 5000
model_epochs: int = 5000

# ============================================================
# Target function
# ============================================================

def target_fn(z):
    base = (
        ((z > -3.5) & (z < -3.0)) |
        ((z > -1.5) & (z < -1.0)) |
        ((z >  0.5) & (z <  1.0)) |
        ((z >  2.5) & (z <  3.0))
    ).float()
    return base + 0.1 * torch.sin(7*z)

# ============================================================
# Adaptive Activation Network (NeuronMLP)
# ============================================================

class NeuronMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, neuron_size),
            nn.SiLU(),
            nn.Linear(neuron_size, neuron_size),
            nn.SiLU(),
            nn.Linear(neuron_size, 1)
        )

    def forward(self, x):
        return self.net(x)

    def pretrain(self, function):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        x_pre = torch.linspace(-5, 5, 4000).unsqueeze(1).to(device)

        for epoch in range(activation_epochs):
            opt.zero_grad()
            y_true = function(x_pre)      # <--- generiek!
            y_pred = self(x_pre)
            loss = F.mse_loss(y_pred, y_true)
            loss.backward()
            opt.step()

            if epoch % 500 == 0:
                print(f"Pretrain epoch {epoch}, loss = {loss.item():.6f}")

# ============================================================
# Wrapper for using activation MLP inside a big network
# ============================================================

class AMLPActivation(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x.unsqueeze(-1)).squeeze(-1)

# ============================================================
# Big MLP (shared architecture for all experiments)
# ============================================================

class BigMLP(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.fc1 = nn.Linear(1, model_size)
        self.fc2 = nn.Linear(model_size, model_size)
        self.fc3 = nn.Linear(model_size, 1)
        self.act = activation

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)

# ============================================================
# Helper: train any model
# ============================================================

def train_model(model, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_train = torch.linspace(-5, 5, 4000).unsqueeze(1).to(device)

    for epoch in range(model_epochs):
        opt.zero_grad()
        y_true = target_fn(x_train)
        y_pred = model(x_train)
        loss = F.mse_loss(y_pred, y_true)
        loss.backward()
        opt.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, loss = {loss.item():.6f}")

    return model

# ============================================================
# EXPERIMENT 1 — Big MLP with static activation
# ============================================================

print("\n=== Training Big MLP with static GELU ===")

model_static = BigMLP(activation=nn.GELU()).to(device)
model_static = train_model(model_static)

# ============================================================
# EXPERIMENT 2 — Big MLP with pretrained GELU activation MLP
# ============================================================

print("\n=== Training Big MLP with trained GELU activation MLP ===")

f_gelu = NeuronMLP().to(device)
f_gelu.pretrain(F.gelu)
model_gelu = BigMLP(activation=AMLPActivation(f_gelu)).to(device)
model_gelu = train_model(model_gelu)

# ============================================================
# EXPERIMENT 3 — Big MLP with random activation MLP
# ============================================================

print("\n=== Training Big MLP with random activation MLP ===")

f_random = NeuronMLP().to(device)
model_random = BigMLP(activation=AMLPActivation(f_random)).to(device)
model_random = train_model(model_random)
    
# ============================================================
# PLOTTING SECTION
# ============================================================

with torch.no_grad():
    x_plot = torch.linspace(-4, 4, 2000).unsqueeze(1).to(device)
    y_true = target_fn(x_plot).cpu()
    y_static = model_static(x_plot).cpu()
    y_gelu = model_gelu(x_plot).cpu()
    y_random = model_random(x_plot).cpu()
    
# ------------------------------------------------------------
# Plot 1: All model predictions vs true function
# ------------------------------------------------------------

plt.figure(figsize=(10,5))
plt.plot(x_plot.cpu(), y_true, label="True function", linewidth=2)
plt.plot(x_plot.cpu(), y_static, label="Static GeLU")
plt.plot(x_plot.cpu(), y_gelu, label="Learned GeLU")
plt.plot(x_plot.cpu(), y_random, label="Learned Random")
plt.title("Comparison of models")
plt.legend()
plt.show()

# ------------------------------------------------------------
# Plot 2: Learned activation vs standard activation
# ------------------------------------------------------------

with torch.no_grad():
    x_act = torch.linspace(-5, 5, 2000).unsqueeze(1).to(device)
    y_default = F.gelu(x_act).cpu()
    y_gelu = f_gelu(x_act).cpu()
    y_random = f_random(x_act).cpu()

plt.figure(figsize=(10,5))
plt.plot(x_act.cpu(), y_default, label="GeLU")
plt.plot(x_act.cpu(), y_gelu, label="Learned activation using GeLU")
plt.plot(x_act.cpu(), y_random, label="Learned activation using random")
plt.title("Comparison of activation functions")
plt.legend()
plt.show()
