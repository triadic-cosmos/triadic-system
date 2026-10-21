from engine.triadic_trainer import TriadicTrainer

MODEL = "mix1"
OLD_PREFIX = "base"
NEW_PREFIX = "small"
NEW_HIDDEN_SIZE = 256

# Scale an existing model to another new model
trainer: TriadicTrainer = TriadicTrainer()

trainer.scale(MODEL, OLD_PREFIX, NEW_PREFIX, NEW_HIDDEN_SIZE)
