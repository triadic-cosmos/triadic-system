# NOT YET UPDATED TO LATEST!

# Output evaluator runner
from engine.triadic_llm import TriadicLLM
from engine.triadic_evaluator import TriadicEvaluator

from dmlg import DATA_FOLDER

import time

DATASET_FOLDER = DATA_FOLDER + "alice/"
EVALUATION_FOLDER = DATASET_FOLDER + "output"
OUTPUT_FILENAME = DATASET_FOLDER + "eval.txt"
MIN_LINES = 20

# Main
llm: TriadicLLM = TriadicLLM()
evaluator: TriadicEvaluator = TriadicEvaluator(llm, MIN_LINES)

start = time.perf_counter()
evaluator.evaluate_folder(EVALUATION_FOLDER, OUTPUT_FILENAME)
print(f"Evaluation time : {time.perf_counter() - start:.1f} s")
