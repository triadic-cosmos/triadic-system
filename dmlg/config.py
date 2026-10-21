# config.py
from dataclasses import dataclass, field
from typing import List

SENTENCE_BASED = True

@dataclass
class Configuration:
    name: str

    # ------------------------------------------------------------
    # GLP model parameters
    # ------------------------------------------------------------
    first_hidden_size: int = 640
    other_hidden_size: int = 640
    lemma_input_dimension: int = 32
    lemma_output_dimension: int = 72
    total_pages: int = 1000000
    max_page_input_size: int = 1
    
    # ------------------------------------------------------------
    # Context parameters
    # ------------------------------------------------------------
    position_divider = 30
    paragraph_divider = 50

    # sentence based input size: 20 x (32 + 72) + 418 + 2 = 2500
    
    narrative_sentences: int = 20
    narrative_state_size: int = 32
    narrative_hidden_size: int = 64
    narrative_token_size: int = 8
    
    tracker_entities: int = 418
    tracker_alpha: float = 0.98

    # not used for sentence based
    
    history_size = []
    history_alpha = []
    
    last_embeddings: int = 5
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
        if SENTENCE_BASED:
            return self.narrative_sentences * self.lemma_output_dimension + \
                   self.narrative_state_size * self.narrative_sentences + \
                   self.tracker_entities + \
                   self.get_cas_vector_size() + 2
        
        # token based encoding
        return self.generator_current_context_size() + \
               self.narrative_state_size * self.narrative_sentences + \
               self.tracker_entities + \
               sum(self.history_size) + \
               self.get_cas_vector_size() + 2 

    def generator_output_size(self) -> int:
        # lemma embedding
        return self.lemma_output_dimension
