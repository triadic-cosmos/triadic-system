from engine.triadic_trainer import TriadicTrainer

import time

MODEL = "mars"
OLD_PREFIX = "300k"
NEW_PREFIX = "large"
NEW_HIDDEN_SIZE = 2048
EPOCHS = 200000

# Scale and continue training an existing model
trainer: TriadicTrainer = TriadicTrainer()

start = time.perf_counter()

trainer.scale(MODEL, OLD_PREFIX, NEW_PREFIX, NEW_HIDDEN_SIZE)

trainer.train(MODEL, NEW_PREFIX, EPOCHS)

print(f"Training time {NEW_PREFIX}: {time.perf_counter() - start:.1f} s")
