# Generate stories with model
from engine.triadic_writer import TriadicWriter
from dmlg import WriterParams

import time

MODEL = "mars"
PREFIXES = ["base_bias"]

NUM_LINES = 3000
MAX_CHAPTERS = 10000
NUM_STORIES = 1
BEAM_SEARCH = False
KEYWORDS = {}

# Quotes from original book
HONEYMOON_PROMPT = [ "Vote for sound men and sound money!" ]
TIME_PROMPT = [ "That is the germ of my great discovery." ]
MARS_PROMPT = [ "Tell me, O Thuvia of Ptarth, that I may still hope, that though you do not love me now, yet some day, some day, my princess." ]

# Generation main using keywords, prompt and beam search
start = time.perf_counter()

print("Generating stories...")

for prefix in PREFIXES:
    writer: TriadicWriter = TriadicWriter(MODEL, prefix)
    params: WriterParams = WriterParams(
        amount = NUM_STORIES,
        lines = NUM_LINES,
        max_chapters = MAX_CHAPTERS,
        prompt = HONEYMOON_PROMPT,
        keywords = KEYWORDS,
        beam_search = BEAM_SEARCH)
    writer.write(params)

print(f"Time: {time.perf_counter() - start:.1f} s")
