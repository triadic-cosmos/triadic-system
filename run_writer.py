# Generate stories with model
from engine.triadic_writer import TriadicWriter
from dmlg import WriterParams

import time

MODEL = "mini"
PREFIXES = ["base"]

NUM_LINES = 1000
MAX_CHAPTERS = 1000
BEAM_SEARCH = False
LEMMA_BLACKLIST = {"mr", "dr"}
PROMPT = []
CAS_VECTOR = [0.9,0.2,0.8,1.0,0.5]
KEYWORDS = {}

# Generation main using keywords, prompt and beam search
start = time.perf_counter()

print("Generating stories...")

for prefix in PREFIXES:
    writer: TriadicWriter = TriadicWriter(MODEL, prefix)
    params: WriterParams = WriterParams(
        lines = NUM_LINES,
        max_chapters = MAX_CHAPTERS,
        prompt = PROMPT,
        keywords = KEYWORDS,
        beam_search = BEAM_SEARCH,
        lemma_blacklist = LEMMA_BLACKLIST,
        cas_vector = CAS_VECTOR        
    )
    writer.write(params)

print(f"Time: {time.perf_counter() - start:.1f} s")
