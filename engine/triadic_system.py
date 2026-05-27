# triadic_system.py
from dataclasses import dataclass, field
from typing import List
import re

from dmlg import (
    WriterAgent,
    MultiAgent,
    Configuration,
    ContextWindow,
    AgentBuilder,
    WriterEnvironment,
    GrammarEngine,
    SemanticEngine,
    Curriculum,
    TokenPage
)

PREFIX = "gen"
MODELS = ["aesop", "forest", "frankenstein", "hyde", "observatory", "poet", "mix", "distill"]
VARIANCE = 0.3
MAX_LINES = 50

@dataclass
class TriadicSystem:
    def __post_init__(self):
        self.story_configuration = Configuration("story")
        self.story_configuration.gen_stories = 1
        self.story_configuration.min_words = 5
        self.story_configuration.max_words = 25
        self.story_configuration.max_tokens = 60
        self.story_configuration.evaluation_threshold = 0.3

        self.builder = AgentBuilder(self.story_configuration)
        self.story_environment = self.builder.build_environment(self.story_configuration, PREFIX)

        # Environments and agents
        self.environments = {}
        self.agent_dict = {}
        for model in MODELS:
            configuration = Configuration(model)
            environment = self.builder.build_environment(configuration, PREFIX)
            self.environments[model] = environment
            self.agent_dict[model] = self.builder.load_or_create_agent(environment)
           
        self.multi_agents: List[MultiAgent] = [
            self.agent_dict["forest"],
            self.agent_dict["observatory"],
            self.agent_dict["poet"],
            self.agent_dict["aesop"],
            self.agent_dict["hyde"]
        ]
        self.multi_weights = [3, 2, 2, 1, 1]

    def print_info(self):
        print("\n\n***** The Triadic Cosmos: DMLG Demo System *****\n")
        print(f"models = {self.agent_dict.keys()}\n")
        print("Welcome to the Dynamic Modular Language Graph demo system!")
        print("You can ask to generate a story with the following prompt :")
        print(f"[1-{MAX_LINES}] lines and model [a model name from above list or multi].")
        print("To repeat the last prompt enter repeat in your prompt.")
        print("To toggle between sampling and beam search mode, enter beam in your prompt.")
        print("To quit enter either stop, quit or exit in your prompt.\n")
        print("Hello, how can I help you?")

    def run_system(self):
        use_beam_search = False
        last_prompt = "Hello!"
        variance = VARIANCE
        story_ctx: ContextWindow = ContextWindow(self.story_configuration)

        while True:
            prompt = input().lower()

            # parse beam search toggle request
            if parse_beam_request(prompt):
                use_beam_search = not use_beam_search
                print(f"Beam search : {use_beam_search}")
                continue

            # parse request to quit
            if parse_quit_request(prompt):
                print("Quitting. Bye!")
                break

            # parse request to repeat
            if parse_repeat_request(prompt):
                print(f"Repeating: {last_prompt}")
                prompt = last_prompt
            else:
                last_prompt = prompt

            # parse request to generate a story
            story_request = parse_story_request(prompt)
            if story_request != None:
                lines = story_request[0]
                name = story_request[1]
                print(f"Generating {lines} lines with model {name}...")
                self.story_configuration.story_lines = lines
                if name == "multi":
                    multi_agent = MultiAgent(self.story_environment, self.multi_agents, self.multi_weights, variance)
                else:           
                    agent = self.agent_dict[name]
                    multi_agent = MultiAgent(self.story_environment, [agent], [10], variance)
                multi_agent.write_story("STORY", story_ctx, None, None, use_beam_search)
                continue

            # fallback reply
            print("How can I help you?")
            

# Check if prompt contains a request to quit
def parse_quit_request(prompt):
    return bool(re.search(r"\b(quit|exit|stop)\b", prompt, re.IGNORECASE))

# Check if prompt contains a request to repeat
def parse_repeat_request(prompt):
    return bool(re.search(r"\brepeat\b", prompt, re.IGNORECASE))

# Check if prompt contains a request to toggle beam search
def parse_beam_request(prompt):
    return bool(re.search(r"\bbeam\b", prompt, re.IGNORECASE))

# Check if prompt contains a request to generate a story
def parse_story_request(text):
    # allowed model names
    models = r"(aesop|forest|frankenstein|hyde|observatory|poet|mix|distill|multi)"
    
    # regex: zoek {number} lines en model {name} in willekeurige volgorde
    pattern = rf"(?i)(?=.*\b(\d+)\s*lines?\b)(?=.*\bmodel\s+{models}\b)"

    # first check if both components exist
    if not re.search(pattern, text):
        return None

    # extract number
    num_match = re.search(r"\b(\d+)\s*lines?\b", text, re.IGNORECASE)
    if not num_match:
        return None
    number = int(num_match.group(1))

    # extract model
    model_match = re.search(rf"\bmodel\s+{models}\b", text, re.IGNORECASE)
    if not model_match:
        return None
    model = model_match.group(1).lower()

    # validate number range
    if not (1 <= number <= MAX_LINES):
        return None

    return number, model
