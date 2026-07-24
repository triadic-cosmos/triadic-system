# config.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class Configuration:
    name: str
    # model parameters
    grammar_hidden_size: int = 256
    lemma_hidden_size: int = 1024
    activation_hidden_size: int = 64
    grammar_dimension: int = 27
    lemma_dimension: int = 256
    total_pages: int = 256
    # learnable lemma embedding rate
    learn_alpha: float = 0.1
    alpha_damping: float = 10
    max_alpha_transitions: int = 5_000_000
    # context
    generator_history_sentences = [5, 15]
    context_max_sentences = 80
    content_max_lemmas = 30
    # curriculum
    no_roundtrip: bool = True
    max_stories: int = 2000
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
    grammar_top_k: int = 3
    lemma_top_k: int = 10
    temperature: float = 0.8
    # beam-search
    nr_of_beams: int = 10
    beam_alpha: float = 0.75
    beam_jitter: float = 0.1
    beam_attempts: int = 3

    def generator_history_context_size(self) -> int:
        return self.generator_history_sentences[0] * 24 + self.generator_history_sentences[1] * 12

    def generator_input_size(self) -> int:
        # narrative memory embedding (128) + sequence embedding (8) + current position (1)
        return self.generator_history_context_size() + self.content_max_lemmas + 137
    
    def generator_output_size(self) -> int:
        return self.total_pages + self.lemma_dimension

    def get_top_k(self, grammar: bool) -> float:
        if grammar:
            return self.grammar_top_k
        return self.lemma_top_k
