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
MODEL_PREFIX = "base"
DATASET_FOLDER = DATA_FOLDER + MODEL
PREFIXES = ["run1"]
MAX_CHAPTERS = 100
MAX_RETRIES = 10
NUM_LINES = 1000
DRAFT = False
PROMPT = True
ITERATIVE = True
LEMMA_BLACKLIST = {}
CAS_VECTORS = [ None ]

WORLD_PROMPT = (
"The main characters are Dr Minsky as chief scientist and Alice as CEO. "
"The company or organization is called Mega Corp. "
"The story revolves around a rapidly growing organization facing a critical transition that may determine its future. "
"The organization's success is built on foundations that may be less stable than they appear. "
"Internal and external groups increasingly question the concentration of influence and decision-making power within the organization. "
"Public trust, transparency, and long-term sustainability are recurring themes. "
"The characters must identify risks, solve emerging problems, and secure the future of the organization. "
"Conflict should emerge from uncertainty, competing interests, and difficult strategic decisions. "
"Add triadic influences using the information, energy, and geometry triad where applicable. "
"Information represents knowledge, transparency, and understanding. "
"Energy represents motivation, ambition, conflict, and change. "
"Geometry represents structure, organization, balance, and concentration of power. "
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
        cas_vector = cas_vector
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
