from engine.triadic_trainer import TriadicTrainer

import time

MODEL = "planet"
WARMUP_EPOCHS = 1
VARIANTS = [50]

# Training a dataset model with epoch variants
trainer: TriadicTrainer = TriadicTrainer()

for variant in VARIANTS:
    start = time.perf_counter()

    prefix = f"{variant}k"
    train_epochs = variant * 1000

    trainer.train(MODEL, prefix, WARMUP_EPOCHS, train_epochs)

    print(f"Training time {prefix}: {time.perf_counter() - start:.1f} s")
