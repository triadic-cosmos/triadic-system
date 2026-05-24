# sentence_encoder.py
from dataclasses import dataclass
from typing import List

from .tokens import Token

LARGE_LENGTH = 10
MEDIUM_LENGTH = 5
LARGE_EMPTY = [Token.NONE.context_embedding] * LARGE_LENGTH
MEDIUM_EMPTY = [Token.NONE.context_embedding] * MEDIUM_LENGTH

@dataclass
class EncodedSentence:
    large: List[Token]
    large_embedding: List[float]
    medium: List[Token]
    medium_embedding: List[float]

EMPTY_SENTENCE = EncodedSentence([], LARGE_EMPTY, [], MEDIUM_EMPTY)

@dataclass
class SentenceEncoder:
    def encode_sentence(self, tokens: List[Token]) -> EncodedSentence:
        noun_verb: List[Token] = []
        adj_adv: List[Token] = []
        adp_pron: List[Token] = []

        prev_tag = None

        for token in tokens:
            if token.is_lemma():
                # noun / verb
                if prev_tag in ("<NOUN>", "<VERB>", "<BASE>"):
                    noun_verb.append(token)

                # adj / adv
                elif prev_tag in ("<ADJ>", "<ADV>"):
                    adj_adv.append(token)

                # adp / pron
                elif prev_tag in ("<ADP>", "<PRON>"):
                    adp_pron.append(token)

            prev_tag = token.text

        # Build large list with priority
        encoded = noun_verb + adj_adv + adp_pron
        encoded = encoded[:LARGE_LENGTH]

        # Pad to LARGE_LENGTH
        if len(encoded) < LARGE_LENGTH:
            encoded += [Token.NONE] * (LARGE_LENGTH - len(encoded))

        large = encoded
        medium = encoded[:MEDIUM_LENGTH]

        return EncodedSentence(
            large, to_embedding(large),
            medium, to_embedding(medium)
        )

def to_embedding(tokens: List[Token]) -> List[float]:
    return [token.context_embedding for token in tokens]
