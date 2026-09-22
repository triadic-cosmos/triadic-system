# Narrator pipeline runner, works on a generated book
from engine.triadic_narrator import TriadicNarrator, TriadicNarratorParams
from engine.triadic_llm import TriadicLLM

DATA_FOLDER = "../triadic-data/toy-system/toy-system-v9/mars"
PREFIXES = ["xl", "xxl_micro", "large1", "base1", "small", "micro", "tiny"]

# Main
llm: TriadicLLM = TriadicLLM()

for prefix in PREFIXES:
    print(f"Processing prefix {prefix}...")
    params: TriadicNarratorParams = TriadicNarratorParams(
        f"{DATA_FOLDER}/output/{prefix}_output.txt",
        f"{DATA_FOLDER}/narrator/{prefix}_book.txt",
        1000)
    narrator: TriadicNarrator = TriadicNarrator(llm, params)
    narrator.process_book()
