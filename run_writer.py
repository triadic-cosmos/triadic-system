# Generate stories with model
from engine.triadic_writer import TriadicWriter

import time

AESOP_KEYWORDS = {
    "fox", "crow", "wolf", "goat", "lion", "mouse",
    "bone", "cliff", "field", "shadow", "water", "stone",
    "watch", "wait", "listen", "carry", "warn", "trade"
}

HYDE_KEYWORDS = {
    "spirit", "hell", "rage", "glee", "body", "blow", "terror", "mist",
    "life", "scene", "excess", "lust", "evil", "gratitude", "remorse",
    "fear", "crime", "ecstasy", "mind", "step", "avenger", "draught",
    "pang", "transformation", "tear", "gratitude", "remorse",
    "veil", "indulgence", "childhood", "memory", "images",
    "sounds", "iniquity", "soul", "joy", "conduct", "existence",
    "humility", "restrictions", "life", "door", "key"
}

OBSERVATORY_KEYWORDS = {
    "observatory", "machine", "ridge", "wind", "dust", "information",
    "caretaker", "dome", "wheel", "panel", "sky", "document",
    "instrument", "schedule", "ledger", "sequence", "lens",
    "horizon", "interval", "vibration", "valley", "mist", "pattern",
    "night", "star", "degree", "instruction", "brightness",
    "shadow", "object", "sunrise", "state", "stillness",
    "measurement", "experiment", "day"
}

POET_KEYWORDS = {"girl", "meadow", "kitten", "rabbit", "echo", "lantern", "hummingbird"}

FOREST_KEYWORDS = {"fox", "rabbit", "raccoon"}

AESOP_PROMPT = [
    "In a quiet field a fox watched a careless crow holding a prize.",
    "A hungry wolf approached a timid goat standing near the edge of a cliff.",
    "The mouse paused when the lion stirred inside the dim cave.",
    "A proud rooster boasted loudly while the farmyard listened in silence.",
    "A clever crow studied the water jar under the heat of the day.",
    "A restless dog guarded a bone while shadows moved around him.",
    "The tortoise walked steadily as the hare laughed at his slow pace.",
    "A thirsty stag lowered his head to the pool and admired his reflection.",
    "A small ant carried a heavy crumb while the grasshopper played nearby.",
    "A young shepherd shouted warnings that no one believed anymore.",
    "A greedy dog saw another dog in the river holding the same treasure.",
    "A traveling merchant listened to the advice of a wise old donkey."
]

OBSERVATORY_PROMPT = ["The observatory stood alone on the ridge."]
FOREST_PROMPT = ["Violent storm tears across hollow forest!"]
HYDE_PROMPT = ["Instantly the spirit of hell awoke in me and raged."]
POET_PROMPT = "The table, the dust, the breath between syllables."

# Generation main using keywords, prompt and beam search
start = time.perf_counter()

print("Generating stories...")
model = "aesop"
writer: TriadicWriter = TriadicWriter(model, 20, 0.3)
writer.write(20, AESOP_PROMPT, AESOP_KEYWORDS, True)

print(f"Time: {time.perf_counter() - start:.1f} s")
