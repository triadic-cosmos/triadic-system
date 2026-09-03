from engine.triadic_trainer import TriadicTrainer

import time

MODEL = "honeymoon"
OLD_PREFIX = "small"
NEW_PREFIX = "xl"
NEW_HIDDEN_SIZE = 3200
EPOCHS = 10000

# Scale and continue training an existing model
trainer: TriadicTrainer = TriadicTrainer()

start = time.perf_counter()

trainer.scale(MODEL, OLD_PREFIX, NEW_PREFIX, NEW_HIDDEN_SIZE)

trainer.train(MODEL, NEW_PREFIX, EPOCHS)

print(f"Training time {NEW_PREFIX}: {time.perf_counter() - start:.1f} s")
