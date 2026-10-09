# Book titler runner
from engine.triadic_llm import TriadicLLM
from engine.triadic_titler import TriadicTitler

from dmlg import DATA_FOLDER

BOOK_PREFIX = "base3"
DATASET_NAME = "alice-hyde-meta"
EXTRA = "_iter"
BOOK_FILENAME = DATA_FOLDER + DATASET_NAME + "/narrator/" + BOOK_PREFIX + EXTRA + "_book.txt"
TITLES = 5

# Main
llm: TriadicLLM = TriadicLLM()
titler: TriadicTitler = TriadicTitler(llm)

titler.title_book(BOOK_FILENAME, TITLES)
