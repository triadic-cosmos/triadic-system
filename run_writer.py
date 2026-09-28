# Generate stories with model
from engine.triadic_writer import TriadicWriter
from dmlg import WriterParams

import time

MODEL = "alice"
PREFIXES = ["base"]

NUM_LINES = 2000
MAX_CHAPTERS = 1000
BEAM_SEARCH = False
KEYWORDS = {}

# Quotes from original book
ALICE_PROMPT = ["The rabbit-hole went straight on like a tunnel."]

# Generation main using keywords, prompt and beam search
start = time.perf_counter()

print("Generating stories...")

for prefix in PREFIXES:
    writer: TriadicWriter = TriadicWriter(MODEL, prefix)
    params: WriterParams = WriterParams(
        lines = NUM_LINES,
        max_chapters = MAX_CHAPTERS,
        prompt = ALICE_PROMPT,
        keywords = KEYWORDS,
        beam_search = BEAM_SEARCH)
    writer.write(params)

print(f"Time: {time.perf_counter() - start:.1f} s")
