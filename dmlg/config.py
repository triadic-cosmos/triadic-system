# config.py
from dataclasses import dataclass
from typing import List

@dataclass
class Configuration:
    name: str
    # model
    hidden_size: int = 8
    activation_hidden_size: int = 32
    max_page_inputs: int = 64
    # context
    generator_history_sentences = [5, 15]
    context_max_sentences = 80
    content_max_lemmas = 30
    # curriculum
    no_roundtrip: bool = True
    max_stories: int = 2000
    min_story_lines: int = 3
    min_sentence_length: int = 10
    epochs_step: int = 10
    # needed epochs depend on dataset size
    explorer_training_epochs: int = 2000 
    generator_training_epochs: int = 1000 
    # generation
    min_words: int = 5
    max_words: int = 25
    max_tokens: int = 60
    story_lines: int = 20
    max_attempts: int = 10000
    score_upper_margin: float = 0.001
    score_lower_margin: float = 0.3
    token_retries: int = 5
    nr_of_beams: int = 5
    max_beams: int = 10
    mark_sentence: bool = False

    def generator_history_context_size(self) -> int:
        return self.generator_history_sentences[0] * 24 + self.generator_history_sentences[1] * 12

    def current_context_size(self) -> int:
        # forelast embedding (4) + last embedding (4) + token index (1) 
        return self.content_max_lemmas + 9

    def generator_input_size(self) -> int:
        # narrative memory embedding (128) + sequence embedding (8) + line index (1)
        return self.generator_history_context_size() + self.current_context_size() + 137
