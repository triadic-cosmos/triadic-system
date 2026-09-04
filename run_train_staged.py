# Staged model training pipeline
from engine.triadic_trainer import TriadicTrainer

import time

HIDDEN_SIZES = [384, 768, 1152, 1536]
EPOCH_MULTIPLIER = [4, 3, 2, 1]

# Run staged training for a model
def run_staged(model: str, prefix: str, total_epochs: int):
    print(f"Running staged training for {model} and {total_epochs} epochs")

    trainer: TriadicTrainer = TriadicTrainer()

    start = time.perf_counter()
    stage_epochs = total_epochs // sum(EPOCH_MULTIPLIER)
    print(f"Stage epochs = {stage_epochs}")

    for index in range(len(HIDDEN_SIZES)):
        stage = f"stage{index+1}"
        epochs = stage_epochs * EPOCH_MULTIPLIER[index]
        hidden_size = HIDDEN_SIZES[index]
        print(f"Running {stage} for {hidden_size} hidden size and {epochs} epochs...")
        
        stage_start = time.perf_counter()
        
        if index == 0:
            # no previous stage, use start baseline model
            trainer.scale(model, prefix, stage, hidden_size)        
        else:
            trainer.scale(model, old_stage, stage, hidden_size)
        
        trainer.train(model, stage, epochs)

        old_stage = stage
        
        print(f"Training time {stage}: {time.perf_counter() - stage_start:.1f} s")

    print(f"Training time: {time.perf_counter() - start:.1f} s")
    print("Finished!")

# Main
run_staged("honeymoon", "test", 200000)
run_staged("time", "test", 100000)
