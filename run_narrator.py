# Narrator pipeline runner, works on a generated draft book or iteratively
from engine.triadic_narrator import (
    TriadicNarrator,
    TriadicIterativeNarrator,
    TriadicPromptNarrator,
    TriadicTransformNarrator,
    TriadicNarratorParams
)
from engine.triadic_writer import TriadicWriter
from engine.triadic_llm import TriadicLLM

from dmlg import WriterParams, DATA_FOLDER

# Parameters
MODELS = ["alice"]
MODEL_PREFIXES = ["base"]
PREFIXES = ["run2"]
CAS_VECTORS = [None]
MAX_CHAPTERS = 100
MAX_RETRIES = 10
DRAFT = False
PROMPT = False
TRANSFORM = True
ITERATIVE = True
LEMMA_BLACKLIST = {}
SUFFIX = "_transform_book" if TRANSFORM else "_iter_book" if PROMPT or ITERATIVE else "_book"
NUM_LINES = 50 if TRANSFORM else 1000

WORLD_PROMPT = (
"The main characters are Dr Minsky as chief scientist and Alice as CEO. "
"The company or organization is called Mega Corp. "
"The story revolves around a rapidly growing organization facing a critical transition that may determine its future. "
"The organization's success is built on foundations that may be less stable than they appear. "
"Internal and external groups increasingly question the concentration of influence and decision-making power within the organization. "
"Public trust, transparency, and long-term sustainability are recurring themes. "
"The characters must identify risks, solve emerging problems, and secure the future of the organization. "
"Conflict should emerge from uncertainty, competing interests, and difficult strategic decisions. "
"Add triadic influences using the knowledge, transformation, and structure triad where applicable. "
)

# Main
llm: TriadicLLM = TriadicLLM()

for index in range(len(PREFIXES)):
    model = MODELS[index]
    model_prefix = MODEL_PREFIXES[index]        
    prefix = PREFIXES[index]
    cas_vector = CAS_VECTORS[index] # put to None if no CAS
    dataset_folder = DATA_FOLDER + model
    print(f"Processing book with prefix {prefix} and CAS {cas_vector}...")

    if not TRANSFORM and (PROMPT or ITERATIVE):
        writer: TriadicWriter = TriadicWriter(model, model_prefix)

    params: TriadicNarratorParams = TriadicNarratorParams(
        input_filename=f"{dataset_folder}/output/{prefix}_output.txt",
        output_filename=f"{dataset_folder}/narrator/{prefix}{SUFFIX}.txt",
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

    # Create the desired narrator
    if TRANSFORM:
        narrator: TriadicTransformNarrator = TriadicTransformNarrator(
            llm, model, params, writer_params)
    elif PROMPT:
        narrator: TriadicPromptNarrator = TriadicPromptNarrator(
            llm, writer, params, writer_params)
    elif ITERATIVE:                
        narrator: TriadicIterativeNarrator = TriadicIterativeNarrator(
            llm, writer, params, writer_params)
    else:
        narrator: TriadicNarrator = TriadicNarrator(llm, params)
    
    narrator.process_book()
