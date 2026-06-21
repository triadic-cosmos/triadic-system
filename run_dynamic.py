# Dynamic learning system runner
from engine.dynamic_system import DynamicSystem, DynamicParams
from engine.triadic_writer import TriadicWriter
from engine.triadic_llm import TriadicLLM

# Parameters
MODEL = "dynamic3"
PREFIX = "comedy"

MODERATE_PROMPT = (
    "Fix the following short story grammatically and semantically. "
    "Make it narratively coherent. "
    "Preserve the core motifs of fox, rabbit, forest, and tension, but allow subtle creative evolution. "
    "Introduce small new imagery, metaphors, or emotional nuances that enrich the canon without breaking it. "
    "Maintain a light comedic tone when appropriate, but allow moments of tension or strangeness. "
    "Respect the fox as generally the hunter and the rabbit as generally the prey, but allow playful inversions if coherent. "
    "Prefer concrete physical action. "    
    "Do not add a title or empty lines. "
    "Use first or third person consistently. "
    "Do not give names to actors. "
    "Avoid using numbers. Avoid complex sentences. "
    "No conversations; use purely descriptive sentences. "
    "End the story with a line containing: The End. "
    "This is the story: "
)

SCORE_PROMPT = (
    "Give a score between 0 and 100 for the following chapter and title. "
    "Evaluate using the following criteria: "
    "coherence, readability, originality, creativity, style, humor, "
    "narrative progression, and suitability as a chapter in a real book. "
    "Higher score for balanced tension and humor; "
    "physical comedy, clumsy movement, chaotic interactions; "  
    "subtle evolution of roles or dynamics; "
    "dynamic scenes with movement or interaction; "
    "imagery or metaphors that enrich the forest world; "
    "creative use of secondary animals. "
    "Lower score for excessive repetition; "
    "static or uneventful scenes; "
    "unclear roles without narrative purpose; "
    "lack of creativity or variation. "
    "Very low score if the story is not coherent or not engaging. "
    "Do not explain the score. "
    "Write the result as: Score: <number> and stop after that. "
    "The chapter title is: $TITLE\n"
    "The chapter text is: $STORY"
)

PARAMS: DynamicParams = DynamicParams(
    PREFIX + "_top",
    PREFIX + "_book",
    PREFIX + "_stats",
    PREFIX + "_exam",
    "The Dark Forest",
    16,
    10,
    20,
    2000,
    85, # 70 without moderation
    92, # 80 without moderation
    65,
    20,
    MODERATE_PROMPT,
    SCORE_PROMPT
)

# Main
llm: TriadicLLM = TriadicLLM()
writer: TriadicWriter = TriadicWriter(MODEL, PREFIX, 15, 0.5)
system: DynamicSystem = DynamicSystem(PARAMS, llm, writer)

if False:
    system.train(500, True)

if True:
    system.examinate(100)
