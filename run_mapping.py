# Mapping demonstration runner
import time

from dmlg import (
    Configuration,
    MappingAgent,
    WriterAgent,
    AgentBuilder,
    WriterEnvironment,
    DATA_FOLDER
)

# Create mapping agent
configuration: Configuration = Configuration("time")
builder: AgentBuilder = AgentBuilder(configuration)
environment: WriterEnvironment = builder.build_environment(configuration, "15k")
writer_agent: WriterAgent = builder.load_or_create_agent(environment)
mapping_agent: MappingAgent = MappingAgent(environment, writer_agent)

# Map input file
start = time.perf_counter()
name = "hyde"
book = "book.txt"
input_filename = DATA_FOLDER + f"{name}/{book}"
output_filename = DATA_FOLDER + f"{name}/{configuration.name}_{book}"
mapping_agent.map_file(input_filename, output_filename)
print(f"Mapping time: {time.perf_counter() - start:.1f} s")
