# Runs a fully recursive lifecycle using boosting
import os
import shutil

from engine.triadic_llm import TriadicLLM
from engine.triadic_booster import TriadicBooster

from dmlg import (
    AgentBuilder,
    WriterAgent,
    WriterEnvironment
)

# Parameters
WARMUP_EPOCHS = 1
RANDOM_EPOCHS = 2000
MODEL_NAME = "planet"
MODEL_PREFIX = "lifecycle"
NR_STORIES = 10
NR_LINES = 20
MIN_LINES = 15
MAX_LINES = 25
RETRIES = 3

# The lifecycle keeps running until stopped
llm: TriadicLLM = TriadicLLM()
booster: TriadicBooster = TriadicBooster(llm, MODEL_NAME, MODEL_PREFIX)
builder: AgentBuilder = booster.builder
agent: WriterAgent = booster.agent
environment: WriterEnvironment = booster.environment
environment.configuration.story_lines = NR_LINES
iteration = 0

while True:
    print(f">>> Starting iteration {iteration} <<<")
    
    # Copy curriculum
    source_curriculum_filename = builder.curriculum_filename(environment, f"book/book-{iteration}")
    curriculum_filename = builder.curriculum_filename(environment, "book")
    preprocessed_filename = builder.preprocessed_filename(environment, "book")
    shutil.copy(source_curriculum_filename, curriculum_filename)

    # Train new model
    curriculum = builder.build_curriculum(environment, "book")
    agent.build_index_from_curriculum(curriculum)
    agent.train_curriculum(curriculum, WARMUP_EPOCHS, RANDOM_EPOCHS)
    target_model_filename = f"{builder.environment_path(environment)}/model/model-{iteration}.bin"
    agent.save(target_model_filename)

    # Delete curriculum
    os.remove(curriculum_filename)
    os.remove(preprocessed_filename)

    # Write some example sampling output
    target_output_filename = builder.curriculum_filename(environment, f"output/output-{iteration}")
    agent.build_output(target_output_filename, NR_STORIES)
    
    # Create new curriculum
    iteration += 1
    target_curriculum_filename = builder.curriculum_filename(environment, f"book/book-{iteration}")    
    booster.boost(target_curriculum_filename, NR_STORIES, NR_LINES, MIN_LINES, MAX_LINES, RETRIES)
