# triadic_writer.py
from dataclasses import dataclass
from typing import List, Set

from dmlg import (
    WriterAgent,
    WriterParams,
    Configuration,
    WriterEnvironment,
    AgentBuilder
)

@dataclass
class TriadicWriter:
    name: str
    prefix: str
    
    def __post_init__(self):
        self.configuration: Configuration = Configuration(self.name)
        self.builder = AgentBuilder(self.configuration)
        self.environment: WriterEnvironment = self.builder.build_environment(self.configuration, self.prefix)
        self.agent: WriterAgent = self.builder.load_or_create_agent(self.environment)
    
    def write(self, params: WriterParams):
        print("Generating output...")

        output_filename = self.builder.output_filename(params, self.agent.environment)
        self.agent.build_output(output_filename, params)
