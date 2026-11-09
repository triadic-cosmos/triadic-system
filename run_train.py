# Train a model
from engine.triadic_trainer import TriadicTrainer

import time

MODEL = "hyde"
PREFIX = "base"
TRAIN = True
CAS_VECTOR_SIZE = 0

# Training a dataset model with epoch variants
start = time.perf_counter()

trainer: TriadicTrainer = TriadicTrainer()

if TRAIN:
    trainer.train(MODEL, PREFIX, CAS_VECTOR_SIZE)
else:
    trainer.evaluate(MODEL, PREFIX, True)

print(f"Training time : {time.perf_counter() - start:.1f} s")
