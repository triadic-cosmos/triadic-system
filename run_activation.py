# Add an activation MLP to a GLPG model to override default SiLU
from engine.triadic_writer import TriadicWriter

from dmlg import ActivationMLP, AMLPActivation, WriterEnvironment, WriterAgent, AgentBuilder

MODEL_NAME = "mars"
INPUT_PREFIX = "base"
OUTPUT_PREFIX = "base_mlp"
HIDDEN_SIZE = 64
EPOCHS = 10000

# Main
writer: TriadicWriter = TriadicWriter(MODEL_NAME, INPUT_PREFIX)
activation: ActivationMLP = ActivationMLP(HIDDEN_SIZE).to_device()
agent: WriterAgent = writer.agent
builder: AgentBuilder = writer.builder

activation.pretrain(EPOCHS)

agent.glp_network.glp_network.act = AMLPActivation(activation)
output_environment: WriterEnvironment = builder.build_environment(
    agent.configuration, OUTPUT_PREFIX)
agent.save(builder.model_filename(output_environment))

activation.plot()
