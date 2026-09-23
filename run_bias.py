# Add a bias MLP to a GLPG model for finetuning
from engine.triadic_writer import TriadicWriter

from dmlg import BiasMLP, AMLPBias, WriterEnvironment, WriterAgent, AgentBuilder

MODEL_NAME = "mars"
INPUT_PREFIX = "base"
OUTPUT_PREFIX = "base_bias"
HIDDEN_SIZE = 16
EPOCHS = 2000

# Main
writer: TriadicWriter = TriadicWriter(MODEL_NAME, INPUT_PREFIX)
agent: WriterAgent = writer.agent
bias: BiasMLP = BiasMLP(agent.configuration.other_hidden_size)
builder: AgentBuilder = writer.builder

bias.pretrain(EPOCHS)

agent.glp_network.glp_network.bias = AMLPBias(bias)
output_environment: WriterEnvironment = builder.build_environment(
    agent.configuration, OUTPUT_PREFIX)
agent.save(builder.model_filename(output_environment))

bias.plot()
