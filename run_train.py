# Train a model
from engine.triadic_trainer import TriadicTrainer

import time

MODEL = "planet"
PREFIX = "base"
EPOCHS = 10000
MAX_EPOCHS = 30000
CAS_VECTOR_SIZE = 0

# Training a dataset model with epoch variants
start = time.perf_counter()

trainer: TriadicTrainer = TriadicTrainer()
trainer.train(MODEL, PREFIX, EPOCHS, MAX_EPOCHS, CAS_VECTOR_SIZE)

print(f"Training time : {time.perf_counter() - start:.1f} s")
