# semantic.py
from typing import List
from dataclasses import dataclass
import string
import re

from .config import Configuration
from .rulebased import BAD_START, BAD_END

VERBOSE = False

@dataclass
class SemanticEngine:
    configuration: Configuration
        
    def _clean(self, sentence: str) -> List[str]:
        cleaned = sentence.strip().translate(str.maketrans("", "", string.punctuation))
        return cleaned.lower().split()

    def validate(self, previous: List[str], sentence: str) -> bool:
        # check if sentence ends with punctuation
        if not (sentence.endswith(".") or sentence.endswith("!") or sentence.endswith("?")):
            print("PUNCTUATION")
            return False
        
        # avoid leftover control tokens
        if sentence.count("<") > 0:
            print("CONTROL")
            return False
                
        words = self._clean(sentence)

        # minimum and maximum number of words
        nr_words = len(words)
        if nr_words < self.configuration.min_words:
            if VERBOSE:
                print("MIN_LENGTH")
            return False
        
        if nr_words > self.configuration.max_words:
            # can happen, this is not logged as error condition
            return False

        # forbidden starts of sentences
        first = words[0]
        if first in BAD_START:
            print("START")
            return False
        
        # forbidden ends of sentences
        last = words[-1]
        if last in BAD_END:
            print("END")
            return False

        # check if sentence is already generated before
        for prev in previous:
            if prev == sentence:
                print("PREVIOUS")
                return False

        # word repeat (at least 2x)
        for i in range(len(words) - 1):
            if words[i] == words[i+1]:
                if VERBOSE:
                    print("A A")
                return False

        # repeat with intermediate word (A B A B)
        for i in range(len(words) - 3):
            if words[i] == words[i+2] and words[i+1] == words[i+3]:
                if VERBOSE:
                    print("A B A B")
                return False

        # trigram repeat (A B C A B C)
        for i in range(len(words) - 5):
            if words[i:i+3] == words[i+3:i+6]:
                print("A B C A B C")
                return False
            
        return True
