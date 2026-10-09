# Narrator pipeline runner, works on a generated draft book or iteratively
from engine.triadic_narrator import TriadicNarrator, TriadicIterativeNarrator, TriadicNarratorParams
from engine.triadic_writer import TriadicWriter
from engine.triadic_llm import TriadicLLM

from dmlg import WriterParams, DATA_FOLDER

MODEL = "alice-hyde-meta"
MODEL_PREFIX = "base"
DATASET_FOLDER = DATA_FOLDER + MODEL
PREFIXES = ["base1", "base2", "base3"]
MAX_CHAPTERS = 100
MAX_RETRIES = 3
NUM_LINES = 2000
ITERATIVE = True
LEMMA_BLACKLIST = {"jekyll", "hyde", "utterson", "poole", "gregor", "mr"}

# Main
llm: TriadicLLM = TriadicLLM()

for prefix in PREFIXES:
    print(f"Processing prefix {prefix}...")
    
    if ITERATIVE:
        writer: TriadicWriter = TriadicWriter(MODEL, MODEL_PREFIX)
        suffix = "_iter_book"
    else:
        suffix = "_book"

    params: TriadicNarratorParams = TriadicNarratorParams(
        input_filename=f"{DATASET_FOLDER}/output/{prefix}_output.txt",
        output_filename=f"{DATASET_FOLDER}/narrator/{prefix}{suffix}.txt",
        max_chapters=MAX_CHAPTERS,
        max_retries=MAX_RETRIES)
    
    writer_params: WriterParams = WriterParams(
        lines = NUM_LINES,
        max_chapters = MAX_CHAPTERS,
        lemma_blacklist = LEMMA_BLACKLIST
    )

    if ITERATIVE:                
        narrator: TriadicIterativeNarrator = TriadicIterativeNarrator(
            llm, writer, params, writer_params)
        narrator.process_book()
    else:
        narrator: TriadicNarrator = TriadicNarrator(llm, params)
        narrator.process_book()
