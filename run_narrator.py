# Narrator pipeline runner, works on a generated draft book or iteratively
from engine.triadic_narrator import (
    TriadicNarrator,
    TriadicIterativeNarrator,
    TriadicPromptNarrator,
    TriadicNarratorParams
)
from engine.triadic_writer import TriadicWriter
from engine.triadic_llm import TriadicLLM

from dmlg import WriterParams, DATA_FOLDER

# Parameters
MODEL = "mini"
MODEL_PREFIX = "token"
DATASET_FOLDER = DATA_FOLDER + MODEL
PREFIXES = ["run10"]
MAX_CHAPTERS = 1000
MAX_RETRIES = 10
NUM_LINES = 2000
DRAFT = True
PROMPT = True
ITERATIVE = False
USE_MATRIX = True
LEMMA_BLACKLIST = {"mr", "dr"}
CAS_VECTORS = [[1.0,0.0,0.0,1.0,0.0]]

WORLD_PROMPT = (
"The main characters are Dr Albert Einstein and Alice. "
"They are scientists who investigate cosmic anomalies. "
"Earth is the primary setting. "
"Any discovered alien world should be called Zazzle. "
"Strong triadic influences should appear naturally throughout the story. "
"A recurring theme is triadic structures and mysterious celestial phenomena. "
"The genre is poetic cosmic science fiction with a subtle horror atmosphere."
)

# Main
llm: TriadicLLM = TriadicLLM()

# Create writer when needed
if PROMPT or ITERATIVE:
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
        world_prompt=WORLD_PROMPT,
        max_chapters=MAX_CHAPTERS,
        max_retries=MAX_RETRIES,
        draft=DRAFT)
    
    writer_params: WriterParams = WriterParams(
        lines = NUM_LINES,
        max_chapters = MAX_CHAPTERS,
        lemma_blacklist = LEMMA_BLACKLIST,
        cas_vector = cas_vector,
        use_matrix = USE_MATRIX
    )

    # Create the desired narrotor
    if PROMPT:
        narrator: TriadicPromptNarrator = TriadicPromptNarrator(
            llm, writer, params, writer_params)
    elif ITERATIVE:                
        narrator: TriadicIterativeNarrator = TriadicIterativeNarrator(
            llm, writer, params, writer_params)
    else:
        narrator: TriadicNarrator = TriadicNarrator(llm, params)
    
    narrator.process_book()
