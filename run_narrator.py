# Narrator pipeline runner
from engine.triadic_narrator import TriadicDistiller, TriadicNarrator, TriadicNarratorParams
from engine.triadic_trainer import TriadicTrainer
from engine.triadic_writer import TriadicWriter
from engine.triadic_llm import TriadicLLM

import time

# Parameters
MIN_CHAPTERS = 4
MAX_CHAPTERS = 20
CANDIDATES = 5
MIN_LINES = 15
RETRIES = 10

# Comedy sequential book parameters
COMEDY_PARAMS = TriadicNarratorParams(
    "comedy",
    "Reject if this is not good material for a slapstick comedy scene.",
    (
        "Combine these sequences into a coherent slapstick comedy scene. "
        "Use clear slapstick beats: setup, misunderstanding, escalation, physical comedy, reversal, punchline. "
        "The environment is dangerous but humorous: unstable ground, falling branches, lava pits, sudden noises. "
        "Keep the danger chaotic but non-lethal. "
        "Use the canonical animals consistently: "
        "fox (cunning but overconfident), rabbit (anxious and reactive), raccoon (wise but sarcastic), monkey (chaotic wildcard). "
        "Maintain fast pacing, visual gags, and physical humor. "
        "Ensure the danger contributes to the comedy and the scene remains coherent."
    ),
    "Find an original way to link the two sequences maintaining the slapstick and danger theme.",
    "This is a slapstick comedy book about animals living in a dangerous environment.",
    "comedy",
    "20k",
    {"fox", "rabbit", "raccoon", "dark", "lava", "lesson", "forest", "shout", "fear"},
    80,
    True
)

# Hyde sequential book parameters
HYDE_PARAMS = TriadicNarratorParams(
    "hyde",
    "Reject if this is not good material for a dark gothic story about internal struggle.",
    ("Combine this into dark gothic story about internal struggle. "
    "This story takes place strictly in the world of Jekyll & Hyde. "
    "Use only the canonical characters Jekyll, Hyde, Utterson, Poole, Bradshaw. "
    "Do not introduce characters or motifs from Frankenstein, Dracula, or other gothic works. "
    "Maintain strict Victorian London atmosphere. "
    "All scientific elements must be consistent with Jekyll’s original experiment. " 
    "All internal struggle belongs to Jekyll and Hyde only. "),
    "Find an original way to link the two sequences maintaining the dark gothic theme.",
    "This is a dark gothic book about internal struggle.",
    "hyde-mix",
    "15k",
    {"shadow", "fog", "laboratory", "potion", "guilt", "stain", "whisper", "conscience"},
    90,
    False
)

# Honeymoon sequential book parameters
HONEYMOON_PARAMS = TriadicNarratorParams(
    "alice",
    None,
    ("Combine this into an amazing space adventure of a man called Lenox and his bride Seraphina. "
     "Only Lenox and Seraphina appear as human characters. "
     "Do not name or describe any other humans. "
     "Polish the semantics where needed."),
    "Link the two sequences maintaining the space adventure theme. Be creative if necessary.",
    "This is a book about a space adventure of a couple on honeymoon.",
    "alice",
    "30k",
    {"orbit","vessel","star","planet","surface","valley","shadow","light","cluster","current","void","wind","look","travel","drift","approach","observe","float","follow","rise","touch","reach","join","meet","embrace"},
    85,
    False
)

# Incremental book parameters
INCREMENTAL_PARAMS = TriadicNarratorParams(
    "mars",
    None,
    ("Rework this draft material into a coherent story, a literary masterpiece. "
    "Make sure there is proper and exciting narrative arc. "
    "Keep all story elements related and realistic. Fill the gaps. "
    "A male main character is called Marvin and a female main character is called Alice. "),
    "Link the sequences maintaining their common theme. Be creative if necessary. ",
    None,
    "mars",
    "50k",
    {},
    85,
    False
)

# Create writer
def create_writer(name: str, prefix: str) -> TriadicWriter:    
    return TriadicWriter(name, prefix, 20)
    
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

# Write a full book with the trained agent
def write_book(params: TriadicNarratorParams):
    llm: TriadicLLM = TriadicLLM()
    narrator: TriadicNarrator = TriadicNarrator(llm, params)
    narrator.write_book(MIN_CHAPTERS, RETRIES)

# Write a sequential narrative book with the trained agent
def write_sequential_book(params: TriadicNarratorParams):
    llm: TriadicLLM = TriadicLLM()
    narrator: TriadicNarrator = TriadicNarrator(llm, params)
    narrator.write_sequential_book(MIN_CHAPTERS, MAX_CHAPTERS, RETRIES)

# Write an incremental story book with the trained agent
def write_incremental_book(params: TriadicNarratorParams):
    llm: TriadicLLM = TriadicLLM()
    narrator: TriadicNarrator = TriadicNarrator(llm, params)
    narrator.write_incremental_book(MIN_CHAPTERS, CANDIDATES, MIN_LINES, RETRIES)

# Main
start = time.perf_counter()
write_incremental_book(INCREMENTAL_PARAMS)
print(f"Elapsed time : {time.perf_counter() - start:.1f} s")
