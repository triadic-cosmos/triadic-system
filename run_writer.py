# Generate stories with model
from engine.triadic_writer import TriadicWriter

import time
 
# Generation main using keywords, prompt and beam search
start = time.perf_counter()

print("Generating stories...")

model = "odyssey"
prefix = "20k"
number_lines = 20
number_stories = 10
beam_search = False
keywords = {}

writer: TriadicWriter = TriadicWriter(model, prefix, number_lines)
writer.write(number_stories, None, keywords, beam_search)

print(f"Time: {time.perf_counter() - start:.1f} s")
