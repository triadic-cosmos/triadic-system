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
    def scale(self, name: str, old_prefix: str, new_prefix: str, new_hidden_size: int):
        configuration = Configuration(name)
        builder = AgentBuilder(configuration)

        old_environment = builder.build_environment(configuration, old_prefix)
        agent: WriterAgent = builder.load_or_create_agent(old_environment)
        if new_hidden_size > agent.glp_network.glp_network.first_hidden_size:
            print("Upscaling agent...")
            new_network = agent.glp_network.glp_network.upscale(new_hidden_size)
        elif new_hidden_size < agent.glp_network.glp_network.first_hidden_size:
            print("Downscaling agent...")
            new_network = agent.glp_network.glp_network.downscale(new_hidden_size)
        agent.glp_network.glp_network = new_network
        
        new_environment = builder.build_environment(configuration, new_prefix)
        agent.save(builder.model_filename(new_environment))
        agent.show()

    def train(self, name: str, prefix: str, random_epochs: int):
        print("Reading curriculum...")
        configuration = Configuration(name)
        configuration.random_epochs = random_epochs
        
        builder = AgentBuilder(configuration)
        environment = builder.build_environment(configuration, prefix)
        curriculum = builder.build_curriculum(environment, "book")
        
        print("Training agent...")
        builder.train_agent(environment, curriculum)
