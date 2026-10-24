# config.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class Configuration:
    name: str

    # ------------------------------------------------------------
    # GLP model parameters
    # ------------------------------------------------------------
    first_hidden_size: int = 768
    other_hidden_size: int = 384
    lemma_input_dimension: int = 64
    lemma_output_dimension: int = 64
    total_pages: int = 1000000
    max_page_input_size: int = 1
    sentence_based: bool = False
    
    # ------------------------------------------------------------
    # Context parameters
    # ------------------------------------------------------------
    position_divider = 30
    paragraph_divider = 50
    
    narrative_sentences: int = 20
    narrative_state_size: int = 32
    narrative_hidden_size: int = 64
    narrative_token_size: int = 8
    
    tracker_entities: int = 585
    tracker_alpha: float = 0.98

    # not used for sentence based mode
    
    history_size = [32, 48, 64]
    history_alpha = [0.92, 0.95, 0.98]
    
    last_embeddings: int = 15
    last_embedding_size: int = 16

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
    learn_alpha: float = 0.001
    random_epochs: int = 1000
    epochs_step: int = 10
    show_epochs_step: int = 100
    
    matrix_positions: int = 10
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
        # current sentence position (1) + punctuation (1)
        return 2 * self.last_embeddings * self.last_embedding_size + 2

    def get_cas_vector_size(self) -> int:
        if hasattr(self, "cas_vector_size"):
            return self.cas_vector_size
        return 0

    def generator_input_size(self) -> int:
        # line position (1) + paragraph position (1) = 2
        
        # sentence based encoding
        if self.sentence_based:
            return self.narrative_sentences * self.lemma_output_dimension + \
                   self.tracker_entities + \
                   self.get_cas_vector_size() + 2
        
        # token based encoding
        return self.generator_current_context_size() + \
               self.narrative_sentences * self.lemma_output_dimension + \
               self.tracker_entities + \
               sum(self.history_size) + \
               self.get_cas_vector_size() + 2 

    def generator_output_size(self) -> int:
        # lemma embedding
        return self.lemma_output_dimension
