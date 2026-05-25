# Shows the non-linear activations learned by the models

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dmlg import (
    TokenPage,
    GrammarEngine,
    SemanticEngine,
    WriterAgent,
    Configuration,
    WriterEnvironment
)

DATA_FOLDER = "../triadic-data/toy-system/"
MODEL_FILENAME = "_model.bin"
MODEL_NAME = "hyde"

def load_agent(name:str, prefix: str) -> WriterAgent:
    return WriterAgent.load(environment, DATA_FOLDER + name + "/" + prefix + MODEL_FILENAME)  
    
import torch
import torch.nn.functional as F
import numpy as np

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
configuration = Configuration(MODEL_NAME)
grammar = GrammarEngine(configuration)
semantic = SemanticEngine(configuration)
environment = WriterEnvironment(configuration, grammar, semantic, "gen")

agent: WriterAgent = load_agent(MODEL_NAME, "gen")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

with torch.no_grad():
    x_act = torch.linspace(-4, 4, 4000).unsqueeze(1).to(device)
    y_default = F.silu(x_act).cpu()
    
    plt.figure(figsize=(10,5))
    plt.plot(x_act.cpu(), y_default, label="SiLU")
    
    index = 1
    
    for pn in agent.paged_network.nets.values():
        if pn is not None and pn.network.act != F.silu:
            page: TokenPage = pn.page
            f = pn.network.act_mlp.to(device)
            y_f = f(x_act).cpu()

            nonlinear, mse = is_nonlinear(x_act, y_f)

            if nonlinear:
                print(page.input_tokens)
                plt.plot(x_act.cpu(), y_f, label=index)
                index += 1

    plt.title("Generator Learned Activation Functions")
    plt.legend()
    plt.show()
