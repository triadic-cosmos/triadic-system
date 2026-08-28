from engine.triadic_trainer import TriadicTrainer

import time

MODEL = "alice"
VARIANTS = [20]

# Training a dataset model with epoch variants
trainer: TriadicTrainer = TriadicTrainer()

for variant in VARIANTS:
    start = time.perf_counter()

    prefix = f"{variant}k"
    train_epochs = variant * 1000

    trainer.train(MODEL, prefix, train_epochs)

    print(f"Training time {prefix}: {time.perf_counter() - start:.1f} s")
