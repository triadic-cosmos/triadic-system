# sentence_encoder.py
from dataclasses import dataclass
from typing import List

from .tokens import Token, LemmaEmbeddingDictionary
from .config import Configuration

NR_TOKEN_SLOTS = 6
TOKEN_SLOTS = {
    "<VERB-PRESENT>": 0,
    "<VERB-PRESENT-3S>": 0,
    "<VERB-PRESENT-1S>": 0,
    "<VERB-PAST>": 0,
    "<VERB-ING>": 0,
    "<VERB-INGV>": 0,
    "<VERB-PERFECT>": 0,

    "<NOUN>": 1,
    "<NOUN-PLURAL>": 1,
    "<PROPN>": 1,

    "<ADJ>": 2,
    "<ADV>": 3,
    "<ADP>": 4,

    "<PRON>": 5,
    "<PRONA>": 5,
}

@dataclass
class EncodedSentence:
    large_embedding: List[float]
    medium_embedding: List[float]

    @staticmethod
    def make_empty_sentence(configuration):
        L = configuration.sentence_large_embedding_size
        M = configuration.sentence_medium_embedding_size
        return EncodedSentence(
            [0.0] * (NR_TOKEN_SLOTS * L),
            [0.0] * (NR_TOKEN_SLOTS * M)
        )

@dataclass
class SentenceEncoder:
    lemma_embedding_dict: LemmaEmbeddingDictionary
    configuration: Configuration

    def empty(self) -> EncodedSentence:
        L = self.configuration.sentence_large_embedding_size
        M = self.configuration.sentence_medium_embedding_size
        return EncodedSentence(
            [0.0] * (NR_TOKEN_SLOTS * L),
            [0.0] * (NR_TOKEN_SLOTS * M)
        )

    def encode_sentence(self, tokens: List[Token]) -> EncodedSentence:
        L = self.configuration.sentence_large_embedding_size
        M = self.configuration.sentence_medium_embedding_size

        large_slots = [0.0] * (NR_TOKEN_SLOTS * L)
        medium_slots = [0.0] * (NR_TOKEN_SLOTS * M)
        slot_counts = [0] * NR_TOKEN_SLOTS

        prev_slot = -1

        for token in tokens:
            if token.is_lemma():
                if prev_slot >= 0:
                    slot_counts[prev_slot] += 1

                    full_emb = self.lemma_embedding_dict.get_input_embedding(token).embedding

                    # CLIP embeddings
                    large_part = full_emb[:L]
                    medium_part = full_emb[L:L+M]

                    baseL = prev_slot * L
                    for i in range(L):
                        large_slots[baseL + i] += large_part[i]

                    baseM = prev_slot * M
                    for i in range(M):
                        medium_slots[baseM + i] += medium_part[i]

                prev_slot = -1

            else:
                prev_slot = TOKEN_SLOTS.get(token.text, -1)

        # normalize
        for slot in range(NR_TOKEN_SLOTS):
            count = slot_counts[slot]
            baseL = slot * L
            baseM = slot * M

            if count > 0:
                for i in range(L):
                    large_slots[baseL + i] /= count
                for i in range(M):
                    medium_slots[baseM + i] /= count

        return EncodedSentence(large_slots, medium_slots)
