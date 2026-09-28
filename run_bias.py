# Add a bias MLP to a GLPG model for finetuning
from engine.triadic_writer import TriadicWriter

from dmlg import WriterEnvironment, WriterAgent, AgentBuilder

MODEL_NAME = "alice"
INPUT_PREFIX = "base"
OUTPUT_PREFIX = "bias"
HIDDEN_SIZE = 256
EPOCHS = 2000

# Main
writer: TriadicWriter = TriadicWriter(MODEL_NAME, INPUT_PREFIX)
agent: WriterAgent = writer.agent
builder: AgentBuilder = writer.builder

agent.add_bias(HIDDEN_SIZE, EPOCHS)

output_environment: WriterEnvironment = builder.build_environment(
    agent.configuration, OUTPUT_PREFIX)
agent.save(builder.model_filename(output_environment))

agent.plot_bias()
