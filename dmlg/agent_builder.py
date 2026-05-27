# agent_builder.py
from dataclasses import dataclass, field
from typing import List
from pathlib import Path

from .grammar import GrammarEngine
from .semantic import SemanticEngine
from .config import Configuration
from .writer_agent import WriterAgent
from .writer_environment import WriterEnvironment
from .multi_agent import MultiAgent
from .curriculum import Curriculum
from .tokens import TokenPage

DATA_FOLDER: str = "../triadic-data/toy-system/"
MODEL_FILENAME: str = "_model.bin"
TOKENS_FILENAME: str = "_tokens.txt"
OUTPUT_FILENAME: str = "_output.txt"

@dataclass
class AgentBuilder:
    configuration: Configuration
    grammar: GrammarEngine = field(init=False)
    semantic: SemanticEngine = field(init=False)
    
    def __post_init__(self):
        self.grammar = GrammarEngine(self.configuration)
        self.semantic = SemanticEngine(self.configuration)
    
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
        curriculum = Curriculum([])
        if Path(preprocessed_filename).is_file():
            curriculum.read_prepocessed(preprocessed_filename, environment.grammar)
            print(f"Read preprocessed curriculum from {preprocessed_filename}.")
        else: 
            curriculum_filename = self.curriculum_filename(environment, name)
            curriculum.read_curriculum(curriculum_filename, environment.grammar, environment.configuration)
            curriculum.write_curriculum(preprocessed_filename)
            print(f"Created curriculum from {curriculum_filename}.")
        print(curriculum)
        return curriculum

    def build_environment(self, configuration: Configuration, prefix: str) -> WriterEnvironment:
        environment = WriterEnvironment(configuration, self.grammar, self.semantic, prefix)
        return environment

    def build_single_agent(self, environment: WriterEnvironment, variance: float) -> MultiAgent:
        agent: WriterAgent = self.load_or_create_agent(environment)
        return MultiAgent(environment, [agent], [10], variance)
                      
    def train_agent(self, environment: WriterEnvironment, curriculum: Curriculum, explore: bool):
        print("Training agent from curriculum...")
        
        if explore:
            epochs = environment.configuration.explorer_training_epochs
        else:
            epochs = environment.configuration.generator_training_epochs
        print(f"epochs = {epochs}")
   
        agent: WriterAgent = self.load_or_create_agent(environment)
        
        if explore:
            agent.build_index_from_curriculum(curriculum)
            print(f"keywords = {len(agent.keyword_map)}")
        
        agent.train_curriculum(curriculum, epochs, explore)        
        agent.save(self.model_filename(environment))

    def optimize_agent(self, environment: WriterEnvironment):
        print("Optimizing agent page models...")                    
        agent: WriterAgent = self.load_or_create_agent(environment)
        agent.optimize()
        agent.save(self.model_filename(environment))
