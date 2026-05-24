# writer_environment.py
from dataclasses import dataclass
from typing import List
import random

from .config import Configuration
from .grammar import GrammarEngine
from .semantic import SemanticEngine

@dataclass
class WriterEnvironment:
    configuration: Configuration
    grammar: GrammarEngine
    semantic: SemanticEngine
    prefix: str
