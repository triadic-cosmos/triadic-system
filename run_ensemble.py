# Generate stories with mix model ensemble
import time

from dmlg import (
    WriterAgent,
    MultiAgent,
    Configuration,
    ContextWindow,
    AgentBuilder,
    WriterEnvironment
)

MODELS = ["5k", "10k", "15k", "20k"]

# Generation main
start = time.perf_counter()

configuration = Configuration("dorian")
builder = AgentBuilder(configuration)
environment = builder.build_environment(configuration, "gen")

# Load agents ensemble
agents = []
path = builder.environment_path(environment)
for model in MODELS:
    print(f"Loading agent {model}.")
    agent = WriterAgent.load(environment, path + model + "_model.bin")
    agents.append(agent)
print(f"agents = {len(agents)}")

multi_agent = MultiAgent(environment, agents, [1] * len(agents), 0.3)
output_filename = builder.curriculum_filename(environment, "ensemble")

# Distillation
print("Starting ensemble distillation...")
stories = 20
lines = 111
    
with open(output_filename, "w", encoding='utf-8-sig') as file:
    for story in range(1, stories + 1):
        print(f"Generating story {story}")
        ctx: ContextWindow = ContextWindow(configuration)        
        line = 1
        sentences = []
        
        while line <= lines:
            line_fraction = (line - 1) / (lines - 1)
            sentence = multi_agent.write_sentence(sentences, ctx, line_fraction)
            print(f"{line}. {sentence}")
            file.write(sentence + "\n")
            line += 1
        
        file.write("\n")
        file.flush()

print(f"Time: {time.perf_counter() - start:.1f} s")
