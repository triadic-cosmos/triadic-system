# config.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class Configuration:
    name: str

    # ------------------------------------------------------------
    # GLP model parameters
    # ------------------------------------------------------------
    first_hidden_size: int = 1024
    other_hidden_size: int = 1024
    lemma_input_dimension: int = 70
    lemma_output_dimension: int = 128
    total_pages: int = 1000000
    max_page_input_size: int = 1
    
    # ------------------------------------------------------------
    # Context parameters
    # ------------------------------------------------------------
    position_divider = 30
    
    # 30 + 50 + 70 = 150
    history_size = [30, 50, 70]
    history_alpha = [0.92, 0.95, 0.98]
    
    # 5 * 32 = 160
    narrative_sentences: int = 5
    narrative_state_size: int = 32
    narrative_hidden_size: int = 64
    narrative_token_size: int = 8
    
    # 2 * 14 * 16 + 2 = 450
    last_embeddings: int = 14
    last_embedding_size: int = 16
    
    # 150 + 160 + 450 + 739 + 1 = 1500
    tracker_entities: int = 739
    tracker_min: int = 10
    tracker_alpha: float = 0.98
    
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
               self.tracker_entities + \
               sum(self.history_size) + 1

    def generator_output_size(self) -> int:
        # lemma embedding
        return self.lemma_output_dimension
