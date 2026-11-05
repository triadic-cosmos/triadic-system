# config.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class Configuration:
    name: str

    # ------------------------------------------------------------
    # GLP model parameters
    # ------------------------------------------------------------
    first_hidden_size: int = 128
    other_hidden_size: int = 128
    token_embedding_dimension: int = 48
    total_pages: int = 1000000
    max_page_input_size: int = 1
    
    # ------------------------------------------------------------
    # Context parameters
    # ------------------------------------------------------------
    position_divider = 30
    paragraph_divider = 50
    
    narrative_sentences: int = 10
    
    tracker_entities: int = 0
    tracker_alpha: float = 0.98
    
    history_size = [48]
    history_alpha = [0.95]
    
    last_embeddings: int = 10

    # ------------------------------------------------------------
    # Curriculum parameters
    # ------------------------------------------------------------
    no_roundtrip: bool = True
    max_sentences: int = 100000
    min_sentence_length: int = 10
    entity_min_count: int = 3

    # ------------------------------------------------------------
    # Training parameters
    # ------------------------------------------------------------
    learn_alpha: float = 0.0
    random_epochs: int = 10000
    show_epochs_step: int = 1000
    matrix_decay: float = 0.99999
    matrix_learning_rate: float = 0.001
    
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
        # current sentence position (1)
        return self.last_embeddings * self.token_embedding_dimension + 1

    def get_cas_vector_size(self) -> int:
        if hasattr(self, "cas_vector_size"):
            return self.cas_vector_size
        return 0

    def generator_sentence_input_size(self) -> int:
        # line position (1) + paragraph position (1) = 2
        return self.narrative_sentences * self.lemma_output_dimension + \
               self.tracker_entities + \
               self.get_cas_vector_size() + 2

    def generator_token_input_size(self) -> int:
        # line position (1) + paragraph position (1) = 2
        return self.generator_current_context_size() + \
               self.narrative_sentences * self.token_embedding_dimension + \
               self.tracker_entities + \
               sum(self.history_size) + \
               self.get_cas_vector_size() + 2

    def generator_output_size(self) -> int:
        # token embedding dimension
        return self.token_embedding_dimension
