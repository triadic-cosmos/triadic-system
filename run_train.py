from engine.triadic_trainer import TriadicTrainer

import time

MODEL = "mini"
EPOCHS = [10000]
MAX_EPOCHS = 50000
CAS_VECTOR_SIZE = 5
SENTENCE = False
PREFIXES = ["token"]

# Training a dataset model with epoch variants
trainer: TriadicTrainer = TriadicTrainer()

for index in range(len(PREFIXES)):
    start = time.perf_counter()

    prefix = PREFIXES[index]
    trainer.train(MODEL, prefix, EPOCHS[index], MAX_EPOCHS, SENTENCE, CAS_VECTOR_SIZE)

    print(f"Training time {prefix}: {time.perf_counter() - start:.1f} s")
