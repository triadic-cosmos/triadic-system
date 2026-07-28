# semantic.py
from typing import List
from dataclasses import dataclass
import string
import re

from .config import Configuration

@dataclass
class SemanticEngine:
    configuration: Configuration
        
    def _clean(self, sentence: str) -> List[str]:
        cleaned = sentence.strip().translate(str.maketrans("", "", string.punctuation))
        return cleaned.lower().split()

    def validate(self, previous: List[str], sentence: str) -> bool:
        # check if sentence ends with punctuation
        if not (sentence.endswith(".") or sentence.endswith("!") or sentence.endswith("?")):
            return False
        
        # avoid leftover control tokens
        if sentence.count("<") > 0:
            return False
        
        words = self._clean(sentence)

        # minimum and maximum number of words
        nr_words = len(words)
        if nr_words < self.configuration.min_words or nr_words > self.configuration.max_words:
            return False

        # forbidden starts of sentences
        first = words[0]
        if first in ["them", "him"]:
            return False

        # check if sentence is already generated before
        for prev in previous:
            if prev == sentence:
                return False

        # word repeat (at least 2x)
        for i in range(len(words) - 1):
            if words[i] == words[i+1]:
                return False

        # repeat with intermediate word (A B A B)
        for i in range(len(words) - 3):
            if words[i] == words[i+2] and words[i+1] == words[i+3]:
                return False

        # trigram repeat (A B C A B C)
        for i in range(len(words) - 5):
            if words[i:i+3] == words[i+3:i+6]:
                return False

        # near-repeats: A B A of A A B
        for i in range(len(words) - 2):
            if words[i] == words[i+2]:
                return False

        # bigram repeat detection
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
        if len(bigrams) != len(set(bigrams)):
            return False

        return True
