from engine.triadic_trainer import TriadicTrainer

import time

# Training a model
start = time.perf_counter()

model = "mix"
prefix = "gen"
explore_epochs = 1000
train_epochs = 1000
explore = True

trainer: TriadicTrainer = TriadicTrainer()
trainer.train(model, prefix, explore_epochs, train_epochs, explore)

print(f"Training time: {time.perf_counter() - start:.1f} s")
