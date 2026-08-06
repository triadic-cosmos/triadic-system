# Builds a booster curriculum from a given model
from engine.triadic_llm import TriadicLLM
from engine.triadic_booster import TriadicBooster

# Parameters
MODEL_NAME = "hyde-boost"
MODEL_PREFIX = "10k"
OUTPUT_NAME = "boost"
EPOCHS = 20
NR_LINES = 15
MIN_LINES = 10
MAX_LINES = 30
RETRIES = 3

# Main keeps running until stopped
llm: TriadicLLM = TriadicLLM()
booster: TriadicBooster = TriadicBooster(llm, MODEL_NAME, MODEL_PREFIX, OUTPUT_NAME)
booster.boost(EPOCHS, NR_LINES, MIN_LINES, MAX_LINES, RETRIES)
