# Shows the non-linear activation learned by the model

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dmlg import (
    TokenPage,
    GrammarEngine,
    SemanticEngine,
    WriterAgent,
    Configuration,
    WriterEnvironment,
    SentenceEncoder
)

DATA_FOLDER = "../triadic-data/toy-system-v5/"
MODEL_FILENAME = "_model.bin"
MODEL_PREFIXES = ["50k"]
MODEL_NAME = "mars"

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
sentence_encoder = SentenceEncoder()
environment = WriterEnvironment(configuration, grammar, semantic, sentence_encoder, "gen")

agents = []
for prefix in MODEL_PREFIXES:
    agent: WriterAgent = load_agent(MODEL_NAME, prefix)
    agents.append(agent)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

with torch.no_grad():
    x_act = torch.linspace(-4, 4, 4000).unsqueeze(1).to(device)
    y_default = F.silu(x_act).cpu()
    
    plt.figure(figsize=(10,5))
    plt.plot(x_act.cpu(), y_default, label="SiLU")
    
    for i, agent in enumerate(agents):
        gf_activation = agent.glp_network.grammar_activation.to(device)
        gy_activation = gf_activation(x_act).cpu()
        plt.plot(x_act.cpu(), gy_activation, label="G:" + MODEL_PREFIXES[i])
        
        lf_activation = agent.glp_network.lemma_activation.to(device)
        ly_activation = lf_activation(x_act).cpu()
        plt.plot(x_act.cpu(), ly_activation, label="L:" + MODEL_PREFIXES[i])

    plt.title("Learned Activation Function")
    plt.legend()
    plt.show()
