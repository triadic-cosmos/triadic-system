# narrative_memory.py
import numpy as np
from typing import List

from .tokens import Token, HISTORY_TOKENS

class NarrativeMemory:
    """
    Narrative memory stores a fixed number of sentence projections.
    New sentences always go into position 0.
    Older sentences shift right: [0] -> [1] -> [2] -> ...
    """

    def __init__(self, lemma_embedding_dict, configuration):
        self.lemma_embedding_dict = lemma_embedding_dict
        self.configuration = configuration

        # Dimensions
        self.S = configuration.narrative_token_size      # token embedding size
        self.H = configuration.narrative_hidden_size     # recurrent hidden size
        self.L = configuration.narrative_state_size      # projection size per sentence
        self.M = configuration.narrative_sentences       # number of stored sentences

        # Full narrative memory vector
        self.narrative_size = self.L * self.M
        self.state = np.zeros(self.narrative_size, dtype=float)

        # Deterministic recurrent projector parameters
        self.W_h = np.eye(self.H) * 0.8
        self.W_e = np.ones((self.H, self.S)) * 0.01
        self.b   = np.zeros(self.H)

        self.V = np.eye(self.L, self.H)[:self.L]
        self.c = np.zeros(self.L)

    def get_state(self) -> List[float]:
        """Return the full narrative memory as a Python list."""
        return self.state.tolist()

    def copy(self) -> "NarrativeMemory":
        copy = NarrativeMemory(self.lemma_embedding_dict, self.configuration)
        copy.state = np.copy(self.state)
        return copy

    # ------------------------------------------------------------
    # UPDATE MEMORY FROM A SENTENCE
    # ------------------------------------------------------------
    def update_from_sentence(self, tokens: List[Token]):
        """
        Extract semantic lemma tokens from the sentence.
        Project them using the recurrent projector.
        Insert the projection at position 0 and shift older entries.
        """

        sentence_vector: List[float] = []
        valid = False

        for tok in tokens:
            t = tok.text

            # Grammar tokens mark that the next lemma is semantically relevant
            if t in HISTORY_TOKENS:
                valid = True
                continue

            # Skip non-lemma tokens
            if not tok.is_lemma():
                continue

            # If grammar token was seen, take this lemma
            if valid:
                valid = False
                full_emb = self.lemma_embedding_dict.get_input_embedding(tok).embedding
                sentence_vector.extend(full_emb[:self.S])

        # No semantic content → nothing to update
        if len(sentence_vector) == 0:
            return

        # Convert to numpy
        embedding = np.array(sentence_vector, dtype=float)

        # Recurrent projection
        P = self.forward(embedding)

        # ------------------------------------------------------------
        # SHIFT MEMORY: newest sentence goes to slot 0
        # ------------------------------------------------------------
        # Shift all existing blocks one position to the right
        if self.M > 1:
            self.state[self.L:] = self.state[:-self.L]

        # Insert new projection at position 0
        self.state[:self.L] = P

    # ------------------------------------------------------------
    # RECURRENT PROJECTOR
    # ------------------------------------------------------------
    def forward(self, sequence: np.ndarray) -> np.ndarray:
        """
        Recurrent embedding over variable-length sequence.
        Sequence is a 1D array containing multiple S-length token embeddings.
        """

        h = np.zeros(self.H, dtype=float)

        # Process sequence in chunks of size S
        for i in range(0, len(sequence), self.S):
            e = sequence[i:i+self.S]
            if len(e) < self.S:
                break

            h = np.tanh(self.W_h @ h + self.W_e @ e + self.b)

        # Final projection
        P = self.V @ h + self.c
        return P
