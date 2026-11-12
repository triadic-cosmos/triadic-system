# Merge existing models into a new model
from engine.triadic_writer import TriadicWriter

from dmlg import (
    Configuration,
    AgentBuilder,
    GlpNetwork,
    WriterAgent,
    WriterEnvironment,
)

MODELS = ["alice", "hyde"]
PREFIXES = ["base"] * len(MODELS)
WEIGHTS = [1.0, 1.0] 

OUTPUT_MODEL = "merge"
OUTPUT_PREFIX = "base"
HIDDEN_SIZE = 192

# Main
writers = []
for index in range(len(MODELS)):
    writers.append(TriadicWriter(MODELS[index], PREFIXES[index]))

networks = [writer.agent.glp_network for writer in writers]

for network in networks:
    network.glp_network = network.glp_network.upscale(HIDDEN_SIZE)

configuration: Configuration = Configuration(OUTPUT_MODEL)
configuration.first_hidden_size = HIDDEN_SIZE
configuration.other_hidden_size = HIDDEN_SIZE

builder: AgentBuilder = AgentBuilder(configuration)
environment: WriterEnvironment = builder.build_environment(configuration, OUTPUT_PREFIX)

merged = builder.load_or_create_agent(environment)
merged.glp_network = GlpNetwork.merge_networks(configuration, networks, WEIGHTS)
merged.save(builder.model_filename(environment))
merged.show(True)
