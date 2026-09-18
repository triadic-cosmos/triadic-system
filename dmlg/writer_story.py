# writer_story.py
from dataclasses import dataclass, field
from typing import List, Set

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

# ------------------------------------------------------------
# Generation parameters
# ------------------------------------------------------------

@dataclass
class WriterParams:
    amount: int = 1
    lines: int = 20
    min_lines_chapter: int = 20
    min_lines_paragraph: int = 3
    min_words: int = 8
    max_words: int = 30
    from_line_fraction: float = 0.0
    to_line_fraction: float = 1.0
    prompt: List[str] = None
    keywords: Set[str] = None
    beam_search: bool = False
    max_tokens: int = 70
    max_attempts: int = 20000
    top_k: int = 50
    temperature: float = 0.001    
    nr_of_beams: int = 3
    beam_alpha: float = 0.8
    beam_jitter: float = 0.5
    beam_attempts: int = 3
    beam_temperature: float = 0.8
    top_boost = [1.3, 1.2, 1.1]
    enable_paging = True
    max_pageless_vocab = 2048
