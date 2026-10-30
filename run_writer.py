# Generate books with model
from engine.triadic_writer import TriadicWriter
from dmlg import WriterParams

import time

MODEL = "planet"
PREFIX = "base"
NUM_LINES = 1000
MAX_CHAPTERS = 100
LEMMA_BLACKLIST = {}
PROMPT = []
CAS_VECTOR = [1] * 0

# Generation main using keywords, prompt and beam search
start = time.perf_counter()

print("Generating books...")

writer: TriadicWriter = TriadicWriter(MODEL, PREFIX)
params: WriterParams = WriterParams(
    lines = NUM_LINES,
    max_chapters = MAX_CHAPTERS,
    prompt = PROMPT,
    lemma_blacklist = LEMMA_BLACKLIST,
    cas_vector = CAS_VECTOR
)
writer.write(params)

print(f"Time: {time.perf_counter() - start:.1f} s")
