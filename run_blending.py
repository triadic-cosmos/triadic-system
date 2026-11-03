# Change model training alpha blending value
from engine.triadic_trainer import TriadicTrainer

MODEL = "planet"
OLD_PREFIX = "base"
NEW_PREFIX = "fix"
ALPHA = -1.0

# Update value within existing model and save to new model
trainer: TriadicTrainer = TriadicTrainer()

trainer.set_blending(MODEL, OLD_PREFIX, NEW_PREFIX, ALPHA)
