# Demonstration of a hybrid LLM x DMLG system
import time

from engine.triadic_llm import TriadicLLM
from engine.triadic_writer import TriadicWriter

from dmlg import (
    ContextWindow,
    WriterStory
)

MODEL = "hyde-mistral"
PREFIX = "15k"
STORIES = 100
LINES = 20
MIN_LINES = 3
VARIANCE = 0.3
BEAM_SEARCH = True
MAX_TOKENS = 1500

# Initialize LLM & DMLG
llm = TriadicLLM()
writer: TriadicWriter = TriadicWriter(MODEL, PREFIX, LINES, VARIANCE)

# Test hybrid story generation
start = time.perf_counter()

output_filename = writer.builder.curriculum_filename(writer.environment, PREFIX + "_mistral_distill")
with open(output_filename, "w", encoding='utf-8-sig') as file:
    for index in range(1, STORIES + 1):
        ctx: ContextWindow = ContextWindow(writer.configuration)
        story: WriterStory = writer.agent.write_story(f"STORY-{index}", ctx, None, None, BEAM_SEARCH)
        writer.agent.fix_story(story)
        moderated = llm.moderate(f"LLM-{index}",story, MAX_TOKENS)
        if len(moderated) >= MIN_LINES:
            for line in moderated:
                file.write(line + "\n")
            file.write("\n")
            file.flush()

print(f"Time: {time.perf_counter() - start:.1f} s")
