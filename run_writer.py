# Generate stories with model
from engine.triadic_writer import TriadicWriter

import time

HYDE_KEYWORDS = {
    "spirit", "hell", "rage", "glee", "body", "blow", "terror", "mist",
    "life", "scene", "excess", "lust", "evil", "gratitude", "remorse",
    "fear", "crime", "ecstasy", "mind", "step", "avenger", "draught",
    "pang", "transformation", "tear", "gratitude", "remorse",
    "veil", "indulgence", "childhood", "memory", "images",
    "sounds", "iniquity", "soul", "joy", "conduct", "existence",
    "humility", "restrictions", "life", "door", "key"
}

TIME_KEYWORDS = {
    "machine", "diagram", "engine", "smoke", "hillside",
    "structure", "gallery", "tremor", "observe", "rise"
}

HYDE_PROMPT = ["Instantly the spirit of hell awoke in me and raged."]

TIME_PROMPT = [
    "A faint mechanical tremor passed through the deserted gallery as the machine waited in the pale haze",
    "He stepped into the dim corridor where ancient diagrams hinted at forgotten futures",
    "The air shimmered with a strange metallic warmth rising from the fractured engine pit",
    "Across the silent hillside the last traces of smoke drifted over abandoned structures",
    "In the trembling light he sensed a presence watching from the edge of the ruined chamber"
]
 
# Generation main using keywords, prompt and beam search
start = time.perf_counter()

print("Generating stories...")

model = "honeymoon"
prefix = "20k"
number_lines = 111
number_stories = 20
variance = 0.3
beam_search = True

writer: TriadicWriter = TriadicWriter(model, prefix, number_lines, variance)
writer.write(number_stories, None, None, beam_search)

print(f"Time: {time.perf_counter() - start:.1f} s")
