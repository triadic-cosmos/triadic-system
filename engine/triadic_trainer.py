# triadic_trainer.py
from dataclasses import dataclass

from dmlg import (
    GrammarEngine,
    SemanticEngine,
    WriterAgent,
    Curriculum,
    WriterEnvironment,
    AgentBuilder,
    Configuration
)

@dataclass
class TriadicTrainer:
    def train(self, name: str, prefix: str, random_epochs: int):
        print("Reading curriculum...")
        configuration = Configuration(name)
        configuration.random_epochs = random_epochs
        
        builder = AgentBuilder(configuration)
        environment = builder.build_environment(configuration, prefix)
        curriculum = builder.build_curriculum(environment, "book")
        
        print("Training agent...")
        builder.train_agent(environment, curriculum)
        