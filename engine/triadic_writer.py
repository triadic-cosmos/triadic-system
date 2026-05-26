# triadic_writer.py
from dataclasses import dataclass
from typing import List, Set

from dmlg import (
    WriterAgent,
    MultiAgent,
    Configuration,
    WriterEnvironment,
    AgentBuilder
)

@dataclass
class TriadicWriter:
    name: str
    prefix: str
    num_lines: int
    variance: float
    
    def __post_init__(self):
        configuration: Configuration = Configuration(self.name)
        configuration.story_lines = self.num_lines
        self.builder = AgentBuilder(configuration)
        environment: WriterEnvironment = self.builder.build_environment(configuration, self.prefix)
        self.agent: MultiAgent = self.builder.build_single_agent(environment, self.variance)
        
    def write(self, amount: int, prompt: List[str] = None, keywords: set[str] = None, beam_search: bool = False):
        print("Generating output...")

        self.agent.build_output(self.builder.output_filename(self.agent.environment), amount, prompt, keywords, beam_search)
