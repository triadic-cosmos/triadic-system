from engine.triadic_trainer import TriadicTrainer

import time

MODEL = "alice"
EPOCHS = [10000]
MAX_EPOCHS = 300000
PREFIXES = ["base"]

# Training a dataset model with epoch variants
trainer: TriadicTrainer = TriadicTrainer()

for index in range(len(PREFIXES)):
    start = time.perf_counter()

    prefix = PREFIXES[index]
    trainer.train(MODEL, prefix, EPOCHS[index], MAX_EPOCHS)

    print(f"Training time {prefix}: {time.perf_counter() - start:.1f} s")
