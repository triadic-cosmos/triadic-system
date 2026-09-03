# Staged model training pipeline
from engine.triadic_trainer import TriadicTrainer

import time

MODEL = "time"
PREFIX = "base"
HIDDEN_SIZES = [160, 320, 640, 1280]
EPOCH_MULTIPLIER = [34, 21, 13, 8]
TOTAL_EPOCHS = 100000

# Scale and continue training an existing model
trainer: TriadicTrainer = TriadicTrainer()

print(f"Running staged training for {MODEL} and {TOTAL_EPOCHS} epochs")

start = time.perf_counter()
stage_epochs = TOTAL_EPOCHS // sum(EPOCH_MULTIPLIER)
print(f"Stage epochs = {stage_epochs}")

for index in range(len(HIDDEN_SIZES)):
    stage = f"stage{index+1}"
    epochs = stage_epochs * EPOCH_MULTIPLIER[index]
    hidden_size = HIDDEN_SIZES[index]
    print(f"Running {stage} for {hidden_size} hidden size and {epochs} epochs...")
    
    stage_start = time.perf_counter()
    
    if index == 0:
        # no previous stage, use start baseline model
        trainer.scale(MODEL, PREFIX, stage, hidden_size)        
    else:
        trainer.scale(MODEL, old_stage, stage, hidden_size)
    
    trainer.train(MODEL, stage, epochs)

    old_stage = stage
    
    print(f"Training time {stage}: {time.perf_counter() - stage_start:.1f} s")

print(f"Training time: {time.perf_counter() - start:.1f} s")
print("Finished!")
