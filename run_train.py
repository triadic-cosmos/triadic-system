from engine.triadic_trainer import TriadicTrainer

import time

# Training a model
start = time.perf_counter()

trainer: TriadicTrainer = TriadicTrainer()
trainer.train("aesop", 10000, 10000, True)

print(f"Training time: {time.perf_counter() - start:.1f} s")
