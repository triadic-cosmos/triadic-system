# config.py
from dataclasses import dataclass, field
from typing import List

NR_TOKENS_SLOTS = 6
TOP_BOOST = [13, 8, 5, 3, 2]

# Paging configuration
ENABLE_PAGING = True
MAX_PAGELESS_VOCAB = 1024

@dataclass
class Configuration:
    name: str

    # ------------------------------------------------------------
    # GLP model parameters
    # ------------------------------------------------------------
    first_hidden_size: int = 1536
    other_hidden_size: int = 768
    lemma_input_dimension: int = 32
    lemma_output_dimension: int = 128
    total_pages: int = 512
    max_page_input_size: int = 16

    # ------------------------------------------------------------
    # Context parameters
    # ------------------------------------------------------------
    generator_history_sentences = [5, 15]
    context_max_sentences = 80
    content_max_lemmas = 30

    # ------------------------------------------------------------
    # Context embeddings
    # ------------------------------------------------------------
    memory_embedding_size: int = 32
    last_embedding_size: int = 16
    sentence_large_embedding_size: int = 4   
    sentence_medium_embedding_size: int = 2
    narrative_state_size: int = 128

    # ------------------------------------------------------------
    # Curriculum parameters
    # ------------------------------------------------------------
    no_roundtrip: bool = True
    max_stories: int = 3000
    min_story_lines: int = 3
    min_sentence_length: int = 10

    # ------------------------------------------------------------
    # Training parameters
    # ------------------------------------------------------------
    learn_alpha: float = 0.001
    random_epochs: int = 1000
    epochs_step: int = 10
    story_prompt: bool = True

    # ------------------------------------------------------------
    # Generation parameters
    # ------------------------------------------------------------
    min_words: int = 5
    max_words: int = 20
    max_tokens: int = 70
    story_lines: int = 20
    max_attempts: int = 10000
    line_divider: float = 20

    # Sampling
    top_k: int = 20
    temperature: float = 0.001
    
    # Beam-search
    nr_of_beams: int = 3
    beam_alpha: float = 0.8
    beam_jitter: float = 0.5
    beam_attempts: int = 3

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
        # current position (2x1) + comma (1)
        return 2 * self.last_embedding_size + \
            2 * self.content_max_lemmas * self.sentence_medium_embedding_size + 3

    def generator_input_size(self) -> int:
        # sequence embedding (8) + line number (1)
        return self.generator_history_context_size() + self.generator_current_context_size() + self.narrative_state_size + 9

    def generator_output_size(self) -> int:
        # lemma embedding
        return self.lemma_output_dimension
