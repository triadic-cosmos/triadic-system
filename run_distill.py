# Model distillation
from dmlg import (
    AgentBuilder,
    MultiAgent,
    WriterAgent,
    Configuration,
    ContextWindow,
    WriterStory
)

# Distillation main
model = "dorian"
prefix = "15k"

configuration: Configuration = Configuration(model)

builder: AgentBuilder = AgentBuilder(Configuration)

environment = builder.build_environment(configuration, prefix)
output_filename = builder.curriculum_filename(environment, prefix + "_distill")
agent = builder.load_or_create_agent(environment)
multi_agent: MultiAgent = MultiAgent(agent.environment, [agent], [10], 0.5)

# Distillation
print("Starting distillation...")
stories = 20
lines = 111
    
with open(output_filename, "w", encoding='utf-8-sig') as file:
    for story in range(1, stories + 1):
        print(f"Generating story {story}")
        ctx: ContextWindow = ContextWindow(configuration)        
        line = 1
        sequence = agent.choose_best_embedding(None)
        sentences = []
        
        while line <= lines:
            line_fraction = (line - 1) / (lines - 1)
            sentence = multi_agent.write_sentence(sentences, ctx, line_fraction)
            print(f"{line}. {sentence}")
            file.write(sentence + "\n")
            line += 1
        
        file.write("\n")
        file.flush()
