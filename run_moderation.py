# Moderated agent demonstration runner 
from typing import List

from dmlg import (
    ModeratedAgent,
    ModeratedSentence,
    AgentBuilder,
    WriterAgent,
    WriterEnvironment,
    Configuration
)

# Configuration
STORIES = 20
LINES = 111
CONFIDENCE = 0.6
MIN_GRAMMAR_VOTES = 1
PREFIXES = ["5k", "10k", "15k", "20k"]

# Create agents for dataset
def create_agents(builder: AgentBuilder, environment: WriterEnvironment) -> List[WriterAgent]:
    path = builder.environment_path(environment)
    agents = []
    for prefix in PREFIXES:
        agent: WriterAgent = WriterAgent.load(environment, path + prefix + "_model.bin")
        agents.append(agent)
    return agents

# Dorian agents
dorian_configuration = Configuration("dorian")
dorian_builder = AgentBuilder(dorian_configuration)
dorian_environment = dorian_builder.build_environment(dorian_configuration, "gen")
dorian_agents = create_agents(dorian_builder, dorian_environment)
dorian_writer_agent = dorian_agents.pop(1) # 10k 

# Dorian moderated agent
dorian_moderation_output_path: str = dorian_builder.curriculum_filename(dorian_environment, "moderated_output")
dorian_moderation_agent: ModeratedAgent = ModeratedAgent(dorian_environment, dorian_writer_agent, dorian_agents)
dorian_moderation_agent.generate_stories(dorian_moderation_output_path, STORIES, LINES, MIN_GRAMMAR_VOTES, CONFIDENCE)
