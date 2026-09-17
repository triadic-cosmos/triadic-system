# config.py
from dataclasses import dataclass, field
from typing import List

NR_TOKENS_SLOTS = 6
TOP_BOOST = [1.3, 1.2, 1.1]

# Paging configuration
ENABLE_PAGING = True
MAX_PAGELESS_VOCAB = 2048

@dataclass
class Configuration:
    name: str

    # ------------------------------------------------------------
    # GLP model parameters
    # ------------------------------------------------------------
    first_hidden_size: int = 1024
    other_hidden_size: int = 1024
    lemma_input_dimension: int = 96
    lemma_output_dimension: int = 128
    total_pages: int = 1000000
    max_page_input_size: int = 1
    
    # ------------------------------------------------------------
    # Context parameters
    # ------------------------------------------------------------
    position_divider = 30
    
    history_size = [8, 16, 32, 48, 64, 80, 96]
    history_alpha = [0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99]
    
    narrative_sentences: int = 20
    narrative_state_size: int = 32
    narrative_hidden_size: int = 64
    narrative_token_size: int = 8
    
    last_embeddings: int = 10
    last_embedding_size: int = 16
    
    # ------------------------------------------------------------
    # Curriculum parameters
    # ------------------------------------------------------------
    no_roundtrip: bool = True
    max_sentences: int = 100000
    min_sentence_length: int = 10

    # ------------------------------------------------------------
    # Training parameters
    # ------------------------------------------------------------
    learn_alpha: float = 0.001
    random_epochs: int = 1000
    epochs_step: int = 10
    show_epochs_step: int = 100

    # ------------------------------------------------------------
    # Generation parameters
    # ------------------------------------------------------------
    min_words: int = 8
    max_words: int = 30
    max_tokens: int = 70
    story_lines: int = 20
    max_attempts: int = 20000

    # Sampling
    top_k: int = 50
    temperature: float = 0.001
    
    # Beam-search
    nr_of_beams: int = 3
    beam_alpha: float = 0.8
    beam_jitter: float = 0.5
    beam_attempts: int = 3
    beam_temperature: float = 0.8
    
    # ------------------------------------------------------------
    # Derived sizes
    # ------------------------------------------------------------
    def generator_history_context_size(self) -> int:
        # uses large + medium embedding sizes
        return NR_TOKENS_SLOTS * (
            self.generator_history_sentences[0] * self.sentence_large_embedding_size +
            self.generator_history_sentences[1] * self.sentence_medium_embedding_size
        )

    def generator_current_context_size(self) -> int:
        # current sentence position (1) + punctuation (1)
        return 2 * self.last_embeddings * self.last_embedding_size + 2

    def generator_input_size(self) -> int:
        # line position (1)
        return self.generator_current_context_size() + \
               self.narrative_state_size * self.narrative_sentences + \
               sum(self.history_size) + 1

    def generator_output_size(self) -> int:
        # lemma embedding
        return self.lemma_output_dimension
