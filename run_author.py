# Autonomous author using DMLG x LLM hybrid intelligence
from engine.triadic_writer import TriadicWriter
from engine.triadic_author import TriadicAuthor
from engine.triadic_llm import TriadicLLM

import time

CHAPTERS = 10
LINES = 20
VARIANCE = 0.3
MODELS = ["comedy"]
PREFIX = ["second"]
HYDE_TIME_TITLE = "Dr. Jekyll and Mr. Hyde Meet The Time Machine"
DORIAN_TIME_TILE = "Dorian Gray meets The Time Machine"
DORIAN_HYDE_TITLE = "Dorian Gray meets Dr. Jekyll and Mr. Hyde"
HYDE_INTRO_FOREST_TITLE = "Dr. Jekyll in the Horror Forest thinking about Language"
INTRO_TITLE = "Introduction to Dynamic Modular Language Graphs"
DORIAN_TITLE = "Picturing Dorian Gray"
OBSERVATORY_TITLE = "The Observatory on the Ridge"
COMEDY_TITLE = "The Fox and the Rabbit"
FILENAME = "../triadic-data/toy-system-v2/author/second_book.txt"

# Create author
llm = TriadicLLM()
writers = []
for i in range(len(MODELS)):
    writer: TriadicWriter = TriadicWriter(MODELS[i], PREFIX[i], LINES, VARIANCE)
    writers.append(writer)
author: TriadicAuthor = TriadicAuthor(llm, writers)

# Write book
start = time.perf_counter()
author.write_book(FILENAME, COMEDY_TITLE, CHAPTERS, LINES)
print(f"Time: {time.perf_counter() - start:.1f} s")
