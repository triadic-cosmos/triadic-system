# writer_story.py
from dataclasses import dataclass, field
from typing import List

from .tokens import Token

@dataclass
class WriterSentence:
    tokens: List[Token]
    natural: str
    fixed: str = field(init=False)
    
    def __post_init__(self):
        self.fixed = self.natural

@dataclass
class WriterStory:
    sentences: List[WriterSentence]

    def get_story(self) -> str:
        joined = " ".join([sentence.fixed for sentence in self.sentences])
        return joined
