# Output evaluator runner
from engine.triadic_llm import TriadicLLM
from engine.triadic_evaluator import TriadicEvaluator

from dmlg import DATA_FOLDER

import time

BOOK_PREFIX = "base3"
DATASET_FOLDER = DATA_FOLDER + "alice-hyde-meta/"
BOOK_FILENAME = DATASET_FOLDER + "output/" + BOOK_PREFIX + "_output.txt"
EVALUATION_FILENAME = DATASET_FOLDER + "evaluation/" + BOOK_PREFIX + "_evaluation.txt"

# Main
llm: TriadicLLM = TriadicLLM()
evaluator: TriadicEvaluator = TriadicEvaluator(llm)

start = time.perf_counter()
evaluator.evaluate_book(BOOK_FILENAME, EVALUATION_FILENAME)
print(f"Evaluation time : {time.perf_counter() - start:.1f} s")
