# narrative_memory.py
import math
import numpy as np
from typing import List, Dict

from .tokens import Token

ACTORS = "actors"
ACTIONS = "actions"
MOMENTS = "moments"
ATMOSPHERE = "atmosphere"

class NarrativeMemory:
    """
    Narrative memory using learned embeddings:
    - tracks actors / actions / moments / atmosphere
    - frequency + recency
    - GRU-like recurrent update
    """

    CATEGORY_SIZES = {
        ACTORS: 60,
        ACTIONS: 30,
        MOMENTS: 20,
        ATMOSPHERE: 10,
    }

    ALPHA = 1.0
    BETA = 1.0
    GAMMA = 0.15

    def __init__(self, lemma_embedding_dict, configuration):
        self.lemma_embedding_dict = lemma_embedding_dict
        self.configuration = configuration

        # persistent narrative state
        self.state = np.zeros(self.configuration.narrative_state_size, dtype=float)

        # category memories: token_text -> {embedding, count, last_seen}
        self.mem = {
            ACTORS: {},
            ACTIONS: {},
            MOMENTS: {},
            ATMOSPHERE: {},
        }

        # GRU-like parameters
        state = self.configuration.narrative_state_size
        cat = self.configuration.memory_embedding_size * 4  # 4 categories

        self.W_f = np.random.randn(state, state + cat) * 0.01
        self.W_u = np.random.randn(state, state + cat) * 0.01
        self.W_s = np.random.randn(state, state + cat) * 0.01

        self.b_f = np.zeros(state)
        self.b_u = np.zeros(state)
        self.b_s = np.zeros(state)

    # ------------------------------------------------------------
    # UPDATE MEMORY FROM A SENTENCE
    # ------------------------------------------------------------
    def update_from_sentence(self, tokens: List[Token]):
        # increment recency
        for cat in self.mem:
            for entry in self.mem[cat].values():
                entry["last_seen"] += 1

        current_category = None
        emb_size = self.configuration.memory_embedding_size

        for tok in tokens:
            t = tok.text

            # grammar tokens determine category
            if t in ["<NOUN>", "<NOUN-PLURAL>", "<PROPN>", "<PRON>", "<PRONA>"]:
                current_category = ACTORS
                continue

            if t in [
                "<VERB-PRESENT>",
                "<VERB-PRESENT-1S>",
                "<VERB-PRESENT-3S>",
                "<VERB-PAST>",
                "<VERB-ING>",
                "<VERB-INGV>",
                "<VERB-PERFECT>",
            ]:
                current_category = ACTIONS
                continue

            if t == "<ADV>":
                current_category = MOMENTS
                continue

            if t in ["<ADJ>"]:
                current_category = ATMOSPHERE
                continue

            # skip non-lemma tokens
            if not tok.is_lemma():
                continue

            # lemma token → add to memory
            if current_category is not None:
                key = tok.text

                full_emb = self.lemma_embedding_dict.get_input_embedding(tok).embedding
                emb = np.array(full_emb[:emb_size], dtype=float)

                if key not in self.mem[current_category]:
                    self.mem[current_category][key] = {
                        "embedding": emb,
                        "count": 1,
                        "last_seen": 0,
                    }
                else:
                    self.mem[current_category][key]["count"] += 1
                    self.mem[current_category][key]["last_seen"] = 0

                current_category = None

        # forget old tokens
        for cat, limit in self.CATEGORY_SIZES.items():
            to_delete = []
            for key, entry in self.mem[cat].items():
                if entry["last_seen"] > limit:
                    to_delete.append(key)
            for key in to_delete:
                del self.mem[cat][key]

        # compute category vectors
        cat_vecs = []
        for cat in [ACTORS, ACTIONS, MOMENTS, ATMOSPHERE]:
            vec = self._compute_category_vector(self.mem[cat])
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            cat_vecs.append(vec)

        combined = np.concatenate(cat_vecs)
        self._update_state(combined)

    # ------------------------------------------------------------
    # CATEGORY VECTOR
    # ------------------------------------------------------------
    def _compute_category_vector(self, entries: Dict[str, dict]):
        size = self.configuration.memory_embedding_size

        if not entries:
            return np.zeros(size)

        weighted = []
        weights = []

        for entry in entries.values():
            count = entry["count"]
            last = entry["last_seen"]
            w = self.ALPHA * count + self.BETA * math.exp(-self.GAMMA * last)
            weighted.append(entry["embedding"] * w)
            weights.append(w)

        total_w = sum(weights)
        if total_w == 0:
            return np.zeros(size)

        return sum(weighted) / total_w

    # ------------------------------------------------------------
    # GRU-LIKE STATE UPDATE
    # ------------------------------------------------------------
    def _update_state(self, combined: np.ndarray):
        x = np.concatenate([self.state, combined])

        f = 1 / (1 + np.exp(-(self.W_f @ x + self.b_f)))
        u = 1 / (1 + np.exp(-(self.W_u @ x + self.b_u)))
        s_tilde = np.tanh(self.W_s @ x + self.b_s)

        self.state = f * self.state + u * s_tilde

    # ------------------------------------------------------------
    # PUBLIC: GET NARRATIVE STATE
    # ------------------------------------------------------------
    def get_state(self) -> List[float]:
        return self.state.tolist()
