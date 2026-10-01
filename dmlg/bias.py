# bias.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Bias MLP
class BiasMLP(nn.Module):
    def __init__(self, h_size, hidden=8):
        super().__init__()
        self.h_size = h_size
        self.hidden = hidden

        # Small modulation MLP
        self.fc1 = nn.Linear(h_size, hidden)
        self.fc2 = nn.Linear(hidden, h_size)

    def forward(self, h):
        """
        h: (batch, h_size)
        return: (batch, h_size)
        """
        x = F.silu(self.fc1(h))
        return self.fc2(x)

    # ---------------------------------------------------------
    # PRETRAIN: learn b(h) ≈ 0 
    # ---------------------------------------------------------
    def pretrain(self, epochs=1000, lr=1e-3):
        opt = torch.optim.Adam(self.parameters(), lr=lr)

        # random hidden states
        x_pre = torch.randn(20000, self.h_size)

        for epoch in range(epochs):
            opt.zero_grad()

            y_true = torch.zeros_like(x_pre)
            y_pred = self(x_pre)

            loss = F.mse_loss(y_pred, y_true)
            loss.backward()
            opt.step()

            if epoch % 200 == 0:
                print(f"[BiasMLP Pretrain] epoch={epoch}, loss={loss.item():.6f}")

    # ---------------------------------------------------------
    # PRETRAIN: learn smooth sinusoidal modulation
    # ---------------------------------------------------------
    def pretrain_sinus(self, epochs=2000, lr=1e-3,
                       amplitude=0.5, frequency=0.02, offset=0.5):
        """
        Train the bias MLP to approximate a smooth sinusoidal modulation
        b(h) = offset + amplitude * sin(frequency * ||h||)

        amplitude: max deviation from offset
        frequency: how fast the sinus oscillates
        offset: baseline modulation level
        """

        opt = torch.optim.Adam(self.parameters(), lr=lr)

        # random hidden states
        x_pre = torch.randn(20000, self.h_size)

        # compute target sinusoidal modulation
        with torch.no_grad():
            norms = x_pre.norm(dim=1)  # (batch,)
            sinus = offset + amplitude * torch.sin(frequency * norms)
            y_true = sinus.unsqueeze(1).repeat(1, self.h_size)

        for epoch in range(epochs):
            opt.zero_grad()

            y_pred = self(x_pre)

            loss = F.mse_loss(y_pred, y_true)
            loss.backward()
            opt.step()

            if epoch % 200 == 0:
                print(f"[BiasMLP Sinus Pretrain] epoch={epoch}, loss={loss.item():.6f}")

    # ---------------------------------------------------------
    # PLOT: bias-norm vs input-norm
    # ---------------------------------------------------------
    def plot(self, samples=4000):
        with torch.no_grad():
            h = torch.randn(samples, self.h_size)
            b = self(h)

            h_norm = h.norm(dim=1).cpu()
            b_norm = b.norm(dim=1).cpu()

        plt.figure(figsize=(10,5))
        plt.scatter(h_norm, b_norm, s=5, alpha=0.3)
        plt.xlabel("||h|| (input norm)")
        plt.ylabel("||b(h)|| (bias norm)")
        plt.title("BiasMLP: Bias Strength vs Input Magnitude")
        plt.show()

# Wrapper for bias MLP
class AMLPBias(nn.Module):
    def __init__(self, bias_mlp):
        super().__init__()
        self.bias_mlp = bias_mlp

    def forward(self, h):
        return self.bias_mlp(h)
