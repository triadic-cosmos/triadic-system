# config.py
from dataclasses import dataclass, field
from typing import List

from .tokens import TokenCodeBook

@dataclass
class Configuration:
    name: str
    # model parameters
    hidden_size: int = 32
    activation_hidden_size: int = 32
    output_dimension: int = 32
    total_pages: int = 300
    # context
    generator_history_sentences = [5, 15]
    context_max_sentences = 80
    content_max_lemmas = 30
    # curriculum
    no_roundtrip: bool = True
    max_stories: int = 1000
    min_story_lines: int = 3
    min_sentence_length: int = 10
    # needed epochs depend on dataset size
    warmup_epochs: int = 1
    random_epochs: int = 1000
    epochs_step: int = 10    
    # generation
    min_words: int = 6
    max_words: int = 25
    max_tokens: int = 60
    story_lines: int = 20
    max_attempts: int = 20000
    # sampling
    top_k: int = 10
    temperature: float = 0.8
    # beam-search
    nr_of_beams: int = 10
    beam_alpha: float = 0.75
    beam_jitter: float = 0.1
    beam_attempts: int = 3

    # code book for output encoding
    codebook: TokenCodeBook = field(init=False)
    
    def __post_init__(self):
        self.codebook = TokenCodeBook(self.output_dimension)

    def generator_history_context_size(self) -> int:
        return self.generator_history_sentences[0] * 24 + self.generator_history_sentences[1] * 12

    def current_context_size(self) -> int:
        # forelast embedding (4) + last embedding (4) + token index (1) 
        return self.content_max_lemmas + 9

    def generator_input_size(self) -> int:
        # narrative memory embedding (128) + sequence embedding (8) + line index (1)
        return self.generator_history_context_size() + self.current_context_size() + 137
