# sentence_encoder.py
from dataclasses import dataclass
from typing import List

from .tokens import Token, LARGE_EMBEDDING_SIZE, MEDIUM_EMBEDDING_SIZE

NR_TOKEN_SLOTS = 6
TOKEN_SLOTS = {
    "<VERB>": 0,
    "<BASE>": 0,
    "<NOUN>": 1,
    "<ADJ>": 2,
    "<ADV>": 3,
    "<ADP>": 4,
    "<PRON>": 5,
}

@dataclass
class EncodedSentence:
    large_embedding: List[float]
    medium_embedding: List[float]


EMPTY_SENTENCE = EncodedSentence(
    Token.NONE.large_embedding * NR_TOKEN_SLOTS,
    Token.NONE.medium_embedding * NR_TOKEN_SLOTS)


@dataclass
class SentenceEncoder:
    def encode_sentence(self, tokens: List[Token]) -> EncodedSentence:
        large_slots = [0.0] * (NR_TOKEN_SLOTS * LARGE_EMBEDDING_SIZE)
        medium_slots = [0.0] * (NR_TOKEN_SLOTS * MEDIUM_EMBEDDING_SIZE)
        slot_counts = [0] * NR_TOKEN_SLOTS

        prev_slot = -1

        for token in tokens:
            if token.is_lemma():
                if prev_slot >= 0:
                    slot_counts[prev_slot] += 1

                    baseL = prev_slot * LARGE_EMBEDDING_SIZE
                    for i, f in enumerate(token.large_embedding):
                        large_slots[baseL + i] += f

                    baseM = prev_slot * MEDIUM_EMBEDDING_SIZE
                    for i, f in enumerate(token.medium_embedding):
                        medium_slots[baseM + i] += f

                prev_slot = -1

            else:
                prev_slot = TOKEN_SLOTS.get(token.text, -1)

        # Normalize or fill with Token.NONE per slot
        for slot in range(NR_TOKEN_SLOTS):
            count = slot_counts[slot]
            baseL = slot * LARGE_EMBEDDING_SIZE
            baseM = slot * MEDIUM_EMBEDDING_SIZE

            if count > 0:
                for i in range(LARGE_EMBEDDING_SIZE):
                    large_slots[baseL + i] /= count
                for i in range(MEDIUM_EMBEDDING_SIZE):
                    medium_slots[baseM + i] /= count
            else:
                # fill with NONE embedding
                for i, f in enumerate(Token.NONE.large_embedding):
                    large_slots[baseL + i] = f
                for i, f in enumerate(Token.NONE.medium_embedding):
                    medium_slots[baseM + i] = f

        return EncodedSentence(large_slots, medium_slots)
