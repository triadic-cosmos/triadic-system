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
    def train(self, name: str, prefix: str, explore_epochs: int, train_epochs: int, explore: bool):
        print("Reading curriculum...")
        configuration = Configuration(name)
        configuration.explorer_training_epochs = explore_epochs
        configuration.generator_training_epochs = train_epochs
        
        builder = AgentBuilder(configuration)
        environment = builder.build_environment(configuration, prefix)
        curriculum = builder.build_curriculum(environment, "book")

        if explore:
            print("Exploring agent...")
            builder.train_agent(environment, curriculum, True)

            print("Optimizing agent...")
            builder.optimize_agent(environment)

        print("Training agent...")
        builder.train_agent(environment, curriculum, False)
