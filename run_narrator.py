# Narrator pipeline runner, works on a generated draft book or iteratively
from engine.triadic_narrator import TriadicNarrator, TriadicIterativeNarrator, TriadicNarratorParams
from engine.triadic_writer import TriadicWriter
from engine.triadic_llm import TriadicLLM

from dmlg import WriterParams, DATA_FOLDER

MODEL = "mini"
MODEL_PREFIX = "base"
DATASET_FOLDER = DATA_FOLDER + MODEL
PREFIXES = ["run3", "run4"]
MAX_CHAPTERS = 1000
MAX_RETRIES = 3
NUM_LINES = 2000
ITERATIVE = True
LEMMA_BLACKLIST = {"mr", "dr"}
CAS_VECTORS = [[0.9,0.2,0.8,1.0,0.5], [0.0,1.5,0.5,0.3,1.0]]

# Main
llm: TriadicLLM = TriadicLLM()

# Create writer when iterative
if ITERATIVE:
    writer: TriadicWriter = TriadicWriter(MODEL, MODEL_PREFIX)
    suffix = "_iter_book"
else:
    suffix = "_book"

for index in range(len(PREFIXES)):
    prefix = PREFIXES[index]
    cas_vector = CAS_VECTORS[index] # put to None if no CAS
    print(f"Processing book with prefix {prefix} and CAS {cas_vector}...")
    
    params: TriadicNarratorParams = TriadicNarratorParams(
        input_filename=f"{DATASET_FOLDER}/output/{prefix}_output.txt",
        output_filename=f"{DATASET_FOLDER}/narrator/{prefix}{suffix}.txt",
        max_chapters=MAX_CHAPTERS,
        max_retries=MAX_RETRIES)
    
    writer_params: WriterParams = WriterParams(
        lines = NUM_LINES,
        max_chapters = MAX_CHAPTERS,
        lemma_blacklist = LEMMA_BLACKLIST,
        cas_vector = cas_vector
    )

    if ITERATIVE:                
        narrator: TriadicIterativeNarrator = TriadicIterativeNarrator(
            llm, writer, params, writer_params)
        narrator.process_book()
    else:
        narrator: TriadicNarrator = TriadicNarrator(llm, params)
        narrator.process_book()
