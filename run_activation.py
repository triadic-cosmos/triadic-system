# Shows the non-linear activation learned by the model

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from dmlg import (
    AgentBuilder,
    TokenPage,
    WriterAgent,
    Configuration,
    WriterEnvironment
)

MODEL_FILENAME = "_model.bin"
MODEL_PREFIXES = ["40k", "50k"]
MODEL_NAMES = ["planet"] * 2

def is_nonlinear(x, y, threshold=0.15):
    x_np = x.cpu().numpy().flatten()
    y_np = y.cpu().numpy().flatten()

    # lineaire regression: y = a*x + b
    A = np.vstack([x_np, np.ones_like(x_np)]).T
    a, b = np.linalg.lstsq(A, y_np, rcond=None)[0]

    # prediction
    y_pred = a * x_np + b

    # mean squared error between real curve and linear fit
    mse = np.mean((y_np - y_pred)**2)

    return mse > threshold, mse

# Main
configuration = Configuration("activation")
builder = AgentBuilder = AgentBuilder(configuration)

agents = []
for index in range(len(MODEL_NAMES)):
    config = Configuration(MODEL_NAMES[index])
    environment = builder.build_environment(config, MODEL_PREFIXES[index])
    agent: WriterAgent = builder.load_or_create_agent(environment)
    agents.append(agent)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

with torch.no_grad():
    x_act = torch.linspace(-4, 4, 4000).unsqueeze(1).to(device)
    y_default = F.silu(x_act).cpu()
    
    plt.figure(figsize=(10,5))
    plt.plot(x_act.cpu(), y_default, label="SiLU")
    
    for i, agent in enumerate(agents):
        f_activation = agent.glp_network.glp_activation.to(device)
        y_activation = f_activation(x_act).cpu()
        plt.plot(x_act.cpu(), y_activation, label=MODEL_PREFIXES[i])
        
    plt.title("Learned Activation Function")
    plt.legend()
    plt.show()
