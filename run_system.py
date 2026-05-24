from engine.triadic_system import TriadicSystem
from engine.triadic_trainer import TriadicTrainer
from engine.triadic_writer import TriadicWriter

import time

# Example keywords and prompts
HYDE_KEYWORDS1 = { "duality", "transformation", "shadow", "laboratory", "potion", "fog", "guilt", "impulse" }
HYDE_KEYWORDS2 = {
    "spirit", "hell", "rage", "glee", "body", "blow", "terror", "mist",
    "life", "scene", "excess", "lust", "evil", "gratitude", "remorse",
    "fear", "crime", "ecstasy", "mind", "step", "avenger", "draught",
    "pang", "transformation", "tear", "gratitude", "remorse",
    "veil", "indulgence", "childhood", "memory", "images",
    "sounds", "iniquity", "soul", "joy", "conduct", "existence",
    "humility", "restrictions", "life", "door", "key"
}
HYDE_PROMPT = ["Instantly the spirit of hell awoke in me and raged."]

OBSERVATORY_KEYWORDS = {
    "observatory", "machine", "ridge", "wind", "dust", "information",
    "caretaker", "dome", "wheel", "panel", "sky", "document",
    "instrument", "schedule", "ledger", "sequence", "lens",
    "horizon", "interval", "vibration", "valley", "mist", "pattern",
    "night", "star", "degree", "instruction", "brightness",
    "shadow", "object", "sunrise", "state", "stillness",
    "measurement", "experiment", "day"
}
OBSERVATORY_PROMPT = ["The observatory stood alone on the ridge."]

POET_KEYWORDS = {"girl", "meadow", "kitten", "rabbit", "echo", "lantern"}
POET_PROMPT = "The table, the dust, the breath between syllables."

MIX_KEYWORDS1 = {"fox", "rabbit", "raccoon"}
MIX_KEYWORDS2 = {"observatory", "star", "caretaker"}
MIX_KEYWORDS3 = {"girl", "kitten", "hummingbird"}


# Triadic System Main
mode = "system"

start = time.perf_counter()

if mode == "system":
    # Main Triadic System Demo
    system: TriadicSystem = TriadicSystem()
    system.print_info()
    system.run_system()
elif mode == "train":
    # Training models
    trainer: TriadicTrainer = TriadicTrainer()
    trainer.train("mix", True)
elif mode == "hyde":
    # Writing stories with hyde model
    print("Generating hyde style text...")
    writer: TriadicWriter = TriadicWriter("hyde", 20, 0.2)
    writer.write(10, HYDE_PROMPT, HYDE_KEYWORDS2, True)
elif mode == "observatory":
    # Writing stories with observatory model
    print("Generating observatory style text...")
    writer: TriadicWriter = TriadicWriter("observatory", 20, 0.3)
    writer.write(10, OBSERVATORY_PROMPT, OBSERVATORY_KEYWORDS, False)
elif mode == "horror":
    # Writing stories with horror model
    print("Generating horror style text...")
    writer: TriadicWriter = TriadicWriter("horror", 111, 0.3)
    writer.write(20)
elif mode == "poet":
    # Writing stories with poet model
    print("Generating poet style text...")
    writer: TriadicWriter = TriadicWriter("poet", 20, 0.5)
    writer.write(20, POET_PROMPT, POET_KEYWORDS, True)
elif mode == "mix":
    # Writing stories with mix model
    print("Generating mix style text...")
    writer: TriadicWriter = TriadicWriter("mix", 20, 0.3)
    writer.write(10, None, MIX_KEYWORDS3, True)

print(f"Time: {time.perf_counter() - start:.3f} s")
