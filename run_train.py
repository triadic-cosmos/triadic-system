from engine.triadic_trainer import TriadicTrainer

import time

# Training a dataset model with epoch variants
model = "honeymoon"
variants = [20]
trainer: TriadicTrainer = TriadicTrainer()

for variant in variants:
    start = time.perf_counter()

    prefix = f"{variant}k"
    explore_epochs = variant * 1000
    train_epochs = variant * 1000
    explore = True

    trainer.train(model, prefix, explore_epochs, train_epochs, explore)

    print(f"Training time {prefix}: {time.perf_counter() - start:.1f} s")
