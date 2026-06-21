# Narrator pipeline runner
from engine.triadic_narrator import TriadicDistiller, TriadicNarrator, TriadicNarratorParams
from engine.triadic_trainer import TriadicTrainer
from engine.triadic_writer import TriadicWriter
from engine.triadic_llm import TriadicLLM

import time

# Parameters
VARIANCE = 0.3
EPOCHS = 15000
STORIES = 100
MIN_CHAPTERS = 5
MAX_CHAPTERS = 10
RETRIES = 3

# Comedy distillation prompt
COMEDY_PROMPT = \
"Write a short comical, absurd story. " + \
"The main actors are a cunning fox and a scared rabbit. Fox is hunter, rabbit is prey." + \
"At most one other animal actor can be in story : a wise raccoon, a crazy monkey or a hyperactive squirrel. " + \
"Do not give names to the animals, just call them fox, raccoon, rabbit, monkey or squirrel, also no capitalisation. " + \
"Location is a dark horror forest that tries to horrify and terrify all its inhabitants. " + \
"Do not use complicated language or sentences. It should be easy to read and funny. " + \
"The story should be at least ten lines and up to 20 total. All sentences have minimum 3 words. " + \
"The story is descriptive, the animals can't talk or do conversations."

# Comedy book parameters
COMEDY_PARAMS = TriadicNarratorParams(
    "comedy",
    "Reject if the sequence breaks the fox and rabbit canon or the dark forest biome. Reject if it has no potential to be funny.",
    "Do not name the actors. Make the story as funny as possible.",
    "Maintain the canon: the fox is the hunter, the rabbit is the prey, and the setting is a dark forest. Preserve the tone: dark forest comedy with light tension and playful mischief.",
    "",
    "comedy",
    ["first", "second", "third", "fourth"],
    ["forest", "mix"],
    ["1k", "1k"],
    {},
    80
)

# Hyde book parameters
HYDE_PARAMS = TriadicNarratorParams(
    "hyde",
    "Reject if the atmosphere is happy or emotionless.",
    "Make the story dark and gothic. Keep conflicting emotions.",
    "Maintain the dark and gothic theme.",
    "",
    "hyde-mistral",
    ["10k", "15k", "20k"],
    ["hyde"],
    ["15k"],
    {},
    60
)

# Owl book parameters
OWL_PARAMS = TriadicNarratorParams(
    "owl",
    "Reject unless this is excellent source material. Reject if there is no cooperation between the birds.",
    "Make it a nice and inspiring story about cooperation, creativity and writing.",
    "Maintain the positive and cooperative atmosphere.",
    "",
    "owl",
    ["10k", "15k"],
    [],
    [],
    {},
    85
)

# Mix book parameters
MIX_PARAMS = TriadicNarratorParams(
    "mix",
    "",
    "Combine all the narrative elements into a hilarious, compelling and outstanding story. Aim for a literary masterpiece.",
    "Maintain the hilarious atmosphere and compelling story.",
    "",
    "mix",
    [],
    # only guest models -> do DMLG blending instead of DMLG ensemble writing
    ["mix", "owl", "comedy", "hyde-mistral"], 
    ["1k", "15k", "third", "10k"],
    {},
    75
)

# Time sequential book parameters
TIME_PARAMS = TriadicNarratorParams(
    "time",
    None,
    "Create a nice retro futuristic book chapter from this material. Use first‑person or third-person only. Smooth semantics where needed. Do not use exact years. Just give the plain text, no titles, sections.",
    "Try to stitch the two given sequences as good as possible, be creative if necessary.",
    "The book is a retro futuristic book with a time machine atmosphere.",
    "time",
    ["10k"], # beam search uses single agent
    [], 
    [],
    {"machine", "engine", "gear", "metal", "light", "shadow", "corridor", "gallery", "structure", "valley", "signal", "tremor", "observe", "shift", "rise", "descend", "fragment", "haze", "surface", "echo"},
    80
)

# Comedy sequential book parameters
COMEDY2_PARAMS = TriadicNarratorParams(
    "comedy",
    None,
    "Create a nice hilarious slapstick chapter from this material. Fox is hunter, rabbit is prey. Use third-person only. Smooth semantics where needed. Just give the plain text, no titles, sections.",
    "Try to stitch the two given sequences as good as possible and keep it light and funny. Be creative where necessary.",
    "The book is dark forest themed animal horror hilarious slapstick comedy.",
    "comedy",
    ["first"], 
    [], 
    [],
    {"fox", "rabbit", "forest", "engine", "metal", "vine", "corridor", "pulse", "lantern", "bark", "signal", "hollow", "hum", "observe", "descend", "coil", "flicker", "resonate", "creep", "distort", "awaken", "shift"},
    80
)

# Hyde sequential book parameters
HYDE2_PARAMS = TriadicNarratorParams(
    "hyde",
    None,
    "Make the story dark and gothic. Keep conflicting emotions. Use third-person only. Smooth semantics where needed. Just give the plain text, no titles, sections.",
    "Maintain the dark and gothic theme during stitching. Be creative where necessary.",
    "The book is dark and gothic with a lot of internal struggle.",
    "hyde-mistral",
    ["10k"],
    [],
    [],
    {"shadow", "breath", "cold", "presence", "entity", "corridor", "engine", "pulse", "hum", "lantern", "metal", "vessel", "echo", "whisper", "tremble", "observe", "descend", "linger", "awaken", "drift"},
    80
)

# Owl sequential book parameters
OWL2_PARAMS = TriadicNarratorParams(
    "owl",
    None,
    "Make it a nice and inspiring story about creativity and mysticism.  Use third-person only. Smooth semantics where needed. Just give the plain text, no titles, sections.",
    "Maintain a positive and mystic atmosphere. Be creative where necessary.",
    "The book is about creativity and mysticism.",
    "owl",
    ["10k"],
    [],
    [],
    {"quill", "spark", "weave", "insight", "pattern", "echo", "guidance", "orbit", "celestial", "horizon", "stillness", "radiance", "alignment", "vastness", "origin", "whisper", "memory", "fragment", "emergence", "continuum", "presence"},
    85
)

# Introduction sequential book parameters
INTRO_PARAMS = TriadicNarratorParams(
    "intro",
    None,
    "Make it a scientific and intriguing piece about language generation.",
    "Maintain a creative and scientific atmosphere about language generation.",
    "This is a book is about language generation.",
    "intro",
    ["1k"],
    [],
    [],
    {"language", "graph", "token", "structure", "model", "generation", "intuition", "production", "navigation", "fog"},
    85
)

# Forest sequential book parameters
FOREST_PARAMS = TriadicNarratorParams(
    "forest",
    None,
    "Combine this into a story about animals in dangerous horror environments.",
    "Find an original way to link the two sequences maintaining the horror theme.",
    "This is a book about animals living in constant horror.",
    "forest",
    ["1k"],
    [],
    [],
    {"fox", "rabbit", "raccoon", "dark", "lava", "lesson", "forest", "shout", "fear"},
    65
)

# Honeymoon sequential book parameters
HONEYMOON_PARAMS = TriadicNarratorParams(
    "honeymoon",
    None,
    "Combine this into an amazing space adventure of a man called Lenox and his bride Seraphina. Only Lenox and Seraphina appear as human characters. Do not name or describe any other humans. Polish the semantics where needed.",
    "Link the two sequences maintaining the space adventure theme. Be creative if necessary.",
    "This is a book about a space adventure.",
    "honeymoon",
    ["20k"],
    [],
    [],
    {"orbit","vessel","star","planet","surface","valley","shadow","light","cluster","current","void","wind","look","travel","drift","approach","observe","float","follow","rise","touch","reach","join","meet","embrace"},
    85
)

# Dynamic book parameters
DYNAMIC_PARAMS = TriadicNarratorParams(
    "comedy",
    "Reject if this story has very little variation.",
    "Combine this into a dark forest slapstick comedy horror story. Polish the semantics where needed.",
    "Link the two sequences maintaining the dark forest slapstick comedy horror theme. Be creative if necessary.",
    "This is a dark forest slapstick comedy horror story book.",
    "dynamic5",
    ["15k"],
    [],
    [],
    {},
    80
)

# Create writer
def create_writer(name: str, prefix: str) -> TriadicWriter:    
    return TriadicWriter(name, prefix, 20, VARIANCE)
    
# Train DMLG agent on curriculum
def train_dmlg(model: str, prefix: str, epochs: int):
    trainer: TriadicTrainer = TriadicTrainer()
    trainer.train(model, prefix, epochs, epochs, True)
        
# Distillation of curriculum from LLM
def distill_llm(model: str, prompt: str, stories: int):
    llm: TriadicLLM = TriadicLLM()
    writer: TriadicWriter = create_writer(model, "dynamic")
    distiller: TriadicDistiller = TriadicDistiller(llm, writer)
    distiller.read_curriculum()
    distiller.distill_llm(prompt, stories)

# Generated moderated distilled curriculum from DMLG agent
def distill_dmlg(model: str, prefix: str, stories: int):
    llm: TriadicLLM = TriadicLLM()    
    writer: TriadicWriter = create_writer(model, prefix)
    distiller: TriadicDistiller = TriadicDistiller(llm, writer)
    distiller.distill_dmlg(stories)

# Write a full book with the trained agents
def write_book(params: TriadicNarratorParams, chapters: int, retries: int):
    llm: TriadicLLM = TriadicLLM()
    narrator: TriadicNarrator = TriadicNarrator(llm, params)
    narrator.write_book(chapters, retries)

# Write a sequential narrative book with the trained agents
def write_sequential_book(params: TriadicNarratorParams, min_chapters: int, max_chapters: int, retries: int):
    llm: TriadicLLM = TriadicLLM()
    narrator: TriadicNarrator = TriadicNarrator(llm, params)
    narrator.write_sequential_book(min_chapters, max_chapters, retries)

# Main
start = time.perf_counter()
write_book(DYNAMIC_PARAMS, MIN_CHAPTERS, RETRIES)
print(f"Elapsed time : {time.perf_counter() - start:.1f} s")
