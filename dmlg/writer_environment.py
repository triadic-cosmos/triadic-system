# writer_environment.py
from dataclasses import dataclass
from typing import List
import random

from .config import Configuration
from .grammar import GrammarEngine
from .semantic import SemanticEngine
from .sentence_encoder import SentenceEncoder

@dataclass
class WriterEnvironment:
    configuration: Configuration
    grammar: GrammarEngine
    semantic: SemanticEngine
    sentence_encoder: SentenceEncoder
    prefix: str
