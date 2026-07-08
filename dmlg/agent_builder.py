# agent_builder.py
from dataclasses import dataclass, field
from typing import List
from pathlib import Path

from .grammar import GrammarEngine
from .semantic import SemanticEngine
from .config import Configuration
from .writer_agent import WriterAgent
from .writer_environment import WriterEnvironment
from .sentence_encoder import SentenceEncoder
from .curriculum import Curriculum
from .tokens import TokenPage

DATA_FOLDER: str = "../triadic-data/toy-system-v4/"
MODEL_FILENAME: str = "_model.bin"
TOKENS_FILENAME: str = "_tokens.txt"
OUTPUT_FILENAME: str = "_output.txt"

@dataclass
class AgentBuilder:
    configuration: Configuration
    grammar: GrammarEngine = field(init=False)
    semantic: SemanticEngine = field(init=False)
    sentence_encoder: SentenceEncoder = field(init=False)
    
    def __post_init__(self):
        self.grammar = GrammarEngine(self.configuration)
        self.semantic = SemanticEngine(self.configuration)
        self.sentence_encoder = SentenceEncoder()
    
    def environment_path(self, environment: WriterEnvironment) -> str:
        return DATA_FOLDER + environment.configuration.name + "/"
        
    def curriculum_filename(self, environment: WriterEnvironment, curriculum: str) -> str:
        return self.environment_path(environment) + curriculum + ".txt"

    def preprocessed_filename(self, environment: WriterEnvironment, curriculum: str) -> str:
        return self.environment_path(environment) + curriculum + TOKENS_FILENAME

    def model_filename(self, environment: WriterEnvironment) -> str:
        return self.environment_path(environment) + environment.prefix + MODEL_FILENAME

    def output_filename(self, environment: WriterEnvironment) -> str:
        return self.environment_path(environment) + environment.prefix + OUTPUT_FILENAME

    def load_or_create_agent(self, environment: WriterEnvironment) -> WriterAgent:
        name = environment.configuration.name
        if Path(self.model_filename(environment)).is_file():
            print(f"Loading existing agent {name}!")
            agent = WriterAgent.load(environment, self.model_filename(environment))
        else:
            print(f"Creating new agent {name}.")
            agent = WriterAgent(environment, name)
        return agent
        
    def build_curriculum(self, environment: WriterEnvironment, name: str) -> Curriculum:
        preprocessed_filename = self.preprocessed_filename(environment, name)
        curriculum = Curriculum()
        if Path(preprocessed_filename).is_file():
            curriculum.read_prepocessed(preprocessed_filename, environment)
            print(f"Read preprocessed curriculum from {preprocessed_filename}.")
        else: 
            curriculum_filename = self.curriculum_filename(environment, name)
            curriculum.read_curriculum(curriculum_filename, environment)
            curriculum.write_curriculum(preprocessed_filename)
            print(f"Created curriculum from {curriculum_filename}.")
        print(curriculum)
        return curriculum

    def build_environment(self, configuration: Configuration, prefix: str) -> WriterEnvironment:
        environment = WriterEnvironment(configuration, self.grammar, self.semantic, self.sentence_encoder, prefix)
        return environment
                      
    def train_agent(self, environment: WriterEnvironment, curriculum: Curriculum):
        print("Training agent from curriculum...")
        
        warmup_epochs = environment.configuration.warmup_epochs
        random_epochs = environment.configuration.random_epochs
        print(f"warmup epochs = {warmup_epochs}")
        print(f"train epochs = {random_epochs}")
   
        agent: WriterAgent = self.load_or_create_agent(environment)
        
        agent.build_index_from_curriculum(curriculum)
        print(f"keywords = {len(agent.keyword_map)}")
        
        agent.train_curriculum(curriculum, warmup_epochs, random_epochs)        
        agent.save(self.model_filename(environment))
