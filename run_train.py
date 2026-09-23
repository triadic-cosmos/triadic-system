from engine.triadic_trainer import TriadicTrainer

import time

MODEL = "mars"
EPOCHS = [100000]
PREFIXES = ["base_bias"]

# Training a dataset model with epoch variants
trainer: TriadicTrainer = TriadicTrainer()

for index in range(len(PREFIXES)):
    start = time.perf_counter()

    prefix = PREFIXES[index]
    trainer.train(MODEL, prefix, EPOCHS[index])

    print(f"Training time {prefix}: {time.perf_counter() - start:.1f} s")
