# Narrator pipeline runner, works on a generated book
from engine.triadic_narrator import TriadicNarrator, TriadicNarratorParams
from engine.triadic_llm import TriadicLLM

from dmlg import DATA_FOLDER

DATASET_FOLDER = DATA_FOLDER + "alice"
PREFIXES = ["bias-1E9", "bias-5", "base", "bias1", "bias0", "bias2", "bias10"]

# Main
llm: TriadicLLM = TriadicLLM()

for prefix in PREFIXES:
    print(f"Processing prefix {prefix}...")
    params: TriadicNarratorParams = TriadicNarratorParams(
        f"{DATASET_FOLDER}/output/{prefix}_output.txt",
        f"{DATASET_FOLDER}/narrator/{prefix}_book.txt",
        1000)
    narrator: TriadicNarrator = TriadicNarrator(llm, params)
    narrator.process_book()
