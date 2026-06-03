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
    Book-driven narrative memory:
    - tracks actors / actions / moments / atmosphere
    - maintains frequency + recency
    - forget timers
    - produces a 128-dim narrative state
    - recurrent GRU-like update
    """

    CATEGORY_SIZES = {
        ACTORS: 60,
        ACTIONS: 30,
        MOMENTS: 20,
        ATMOSPHERE: 10,
    }

    # hyperparameters for weighting
    ALPHA = 1.0     # frequency weight
    BETA = 1.0      # recency weight
    GAMMA = 0.15    # recency decay

    def __init__(self):
        # persistent state (128 floats)
        self.state = np.zeros(128, dtype=float)

        # category memories: token_text -> {embedding, count, last_seen}
        self.mem = {
            ACTORS: {},
            ACTIONS: {},
            MOMENTS: {},
            ATMOSPHERE: {},
        }

        # GRU-like parameters
        self.W_f = np.random.randn(128, 256) * 0.01
        self.W_u = np.random.randn(128, 256) * 0.01
        self.W_s = np.random.randn(128, 256) * 0.01

        self.b_f = np.zeros(128)
        self.b_u = np.zeros(128)
        self.b_s = np.zeros(128)

    # ------------------------------------------------------------
    # UPDATE MEMORY FROM A SENTENCE
    # ------------------------------------------------------------
    def update_from_sentence(self, tokens: List[Token]):
        # increment recency
        for cat in self.mem:
            for entry in self.mem[cat].values():
                entry["last_seen"] += 1

        current_category = None

        for tok in tokens:
            t = tok.text

            # 1. Grammar tokens determine category
            if t in ["<NOUN>", "<PROPN>", "<PRON>"]:
                current_category = ACTORS
                continue

            if t in ["<VERB>", "<BASE>"]:
                current_category = ACTIONS
                continue

            # ONLY <ADV> is a category grammar token for MOMENTS
            if t == "<ADV>":
                current_category = MOMENTS
                continue

            # <ADJ> and <ING> define atmosphere
            if t in ["<ADJ>", "<ING>"]:
                current_category = ATMOSPHERE
                continue

            # 2. If it's not a lemma, skip it (postfix tokens also land here)
            if not tok.is_lemma():
                continue

            # 3. Lemma token: add to memory
            if current_category is not None:
                key = tok.text

                if key not in self.mem[current_category]:
                    self.mem[current_category][key] = {
                        "embedding": np.array(tok.memory_embedding, dtype=float),
                        "count": 1,
                        "last_seen": 0,
                    }
                else:
                    self.mem[current_category][key]["count"] += 1
                    self.mem[current_category][key]["last_seen"] = 0

                # after a lemma, reset category
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
            # normalize to avoid category dominance
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            cat_vecs.append(vec)

        # concatenate → 128-dim
        combined = np.concatenate(cat_vecs)

        # update persistent state
        self._update_state(combined)

    # ------------------------------------------------------------
    # CATEGORY VECTOR
    # ------------------------------------------------------------
    def _compute_category_vector(self, entries: Dict[str, dict]):
        if not entries:
            return np.zeros(32)

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
            return np.zeros(32)

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
