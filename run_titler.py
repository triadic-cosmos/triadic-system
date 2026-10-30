# Generate titles for a narrator book
from engine.triadic_llm import TriadicLLM
from engine.triadic_titler import TriadicTitler

from dmlg import DATA_FOLDER

DATASET_NAME = "alice"
BOOK_PREFIX = "run1"
EXTRA = "_iter"
BOOK_FILENAME = DATA_FOLDER + DATASET_NAME + "/narrator/" + BOOK_PREFIX + EXTRA + "_book.txt"
TITLES = 5

# Main
llm: TriadicLLM = TriadicLLM()
titler: TriadicTitler = TriadicTitler(llm)

titler.title_book(BOOK_FILENAME, TITLES)
