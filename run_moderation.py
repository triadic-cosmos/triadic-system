# Moderated agent demonstration runner 
from dmlg import (
    ModeratedAgent,
    ModeratedSentence,
    AgentBuilder,
    WriterAgent,
    Configuration
)

# Configuration
STORIES = 10
LINES = 20
RUN_WRITER = True
RUN_LEAKYRELU = False
RUN_HARDTANH = False
RUN_SILUGELU = False
WRITER_CONFIDENCE = 0.5 # different datasets, lower threshold
TIME_CONFIDENCE = 0.8 # same datasets, higher threshold
WRITER_MODEL = "frankenstein"

# Frankenstein writer agent
writer_configuration = Configuration(WRITER_MODEL)
writer_builder = AgentBuilder(writer_configuration)
writer_environment = writer_builder.build_environment(writer_configuration, "gen")
writer_agent: WriterAgent = writer_builder.load_or_create_agent(writer_environment)

# Time Machine moderation
time_configuration = Configuration("time")
time_builder = AgentBuilder(time_configuration)
time_environment = time_builder.build_environment(time_configuration, "gen")
path = time_builder.environment_path(time_environment)

# Moderator trained time agents with different activation functions
silu_time_agent: WriterAgent = WriterAgent.load(time_environment, path + "silu_model.bin")
gelu_time_agent: WriterAgent = WriterAgent.load(time_environment, path + "gelu_model.bin")
silugelu_time_agent: WriterAgent = WriterAgent.load(time_environment, path + "silugelu_model.bin")
moderators = [silu_time_agent, gelu_time_agent, silugelu_time_agent]

# Undertrained time agents with bad activation functions
leakyrelu_time_agent: WriterAgent = WriterAgent.load(time_environment, path + "leakyrelu_model.bin")
hardtanh_time_agent: WriterAgent = WriterAgent.load(time_environment, path + "hardtanh_model.bin")

# Undertrained hardtanh time agent generation with moderation of time agents
if RUN_LEAKYRELU:
    moderation_output_path: str = writer_builder.curriculum_filename(time_environment, "leakyrelu_moderated_output")
    leakyrelu_moderation_agent: ModeratedAgent = ModeratedAgent(time_environment, leakyrelu_time_agent, moderators)
    leakyrelu_moderation_agent.generate_stories(moderation_output_path, STORIES, LINES, TIME_CONFIDENCE)

# Undertrained hardtanh time agent generation with moderation of time agents
if RUN_HARDTANH:
    moderation_output_path: str = writer_builder.curriculum_filename(time_environment, "hardtanh_moderated_output")
    hardtanh_moderation_agent: ModeratedAgent = ModeratedAgent(time_environment, hardtanh_time_agent, moderators)
    hardtanh_moderation_agent.generate_stories(moderation_output_path, STORIES, LINES, TIME_CONFIDENCE)

# Trained silugelu time agent generation with moderation of other time agents
if RUN_SILUGELU:
    other_moderators = [leakyrelu_time_agent, silu_time_agent, gelu_time_agent]
    moderation_output_path: str = writer_builder.curriculum_filename(time_environment, "silugelu_moderated_output")
    silugelu_moderation_agent: ModeratedAgent = ModeratedAgent(time_environment, silugelu_time_agent, other_moderators)
    silugelu_moderation_agent.generate_stories(moderation_output_path, STORIES, LINES, TIME_CONFIDENCE)

# Writer agent generation with moderation of time agents
if RUN_WRITER:
    moderation_output_path: str = writer_builder.curriculum_filename(writer_environment, "moderated_output")
    moderated_agent: ModeratedAgent = ModeratedAgent(writer_environment, writer_agent, moderators)
    moderated_agent.generate_stories(moderation_output_path, STORIES, LINES, WRITER_CONFIDENCE)
