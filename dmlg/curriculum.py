# curriculum.py
from dataclasses import dataclass, field
from typing import List
import hashlib
import struct
import random
import re

from .tokens import Token, TokenDictionary
from .grammar import GrammarEngine
from .config import Configuration
from .writer_environment import WriterEnvironment

VERBOSE = False

@dataclass
class CurriculumSentence:
    tokens: List[Token]
    natural: str

    def get_canonical(self) -> str:
        return " ".join([token.text for token in self.tokens])

@dataclass(frozen=True)
class Curriculum:
    sentences: List[CurriculumSentence] = field(default_factory=list)
    token_dictionary: TokenDictionary = field(default_factory=TokenDictionary) 

    def get_random_sentence(self, rng:random.Random) -> CurriculumSentence:
        return self.sentences[rng.randrange(0, len(self.sentences))]

    # ============================================================
    # Curriculum reader
    # ============================================================

    def add_to_curriculum(self, start_token: str, sentences: List[str], environment: WriterEnvironment):
        combined: str = " ".join(sentences)
        split = re.split("<SPLIT>", combined)
        
        i = 1
        for s in split:
            if len(s) < environment.configuration.min_sentence_length:
                continue
            if not s.endswith("<PERIOD>") and not s.endswith("<EXCLAMATION>") and not s.endswith("<QUESTION>"):
                continue
            
            eol = s.replace("<COMMA> <PERIOD>", "<PERIOD>") + " <EOL>"
            if i == 1 and start_token:
                eol = start_token + eol
            tokens = [self.token_dictionary.add_and_get(t) for t in eol.split(" ") if t != ""]
            natural = environment.grammar.convert_from_canonical(eol)
            self.sentences.append(CurriculumSentence(tokens, natural))
            if VERBOSE:
                print(f"{i}. {natural}")
            i += 1
            if len(self.sentences) >= environment.configuration.max_sentences:
                break

    def read_curriculum(self, filename: str, environment: WriterEnvironment):
        with open(filename, "r", encoding='utf-8-sig') as f:
            lines = f.read().splitlines()

        chapter_nr = 1
        sentences = []
        start_token = "<SOC> "

        for line in lines:
            line = preprocess_line(line)
            if len(line) < 5:
                if not start_token:
                    start_token = "<SOP> "
                if len(sentences) > 0:
                    self.add_to_curriculum(start_token, sentences, environment)
                    start_token = None
                    sentences = []
                    if len(self.sentences) >= environment.configuration.max_sentences:
                        break
                    continue
                
            doc = environment.grammar.nlp(line)
            
            for sent in doc.sents:
                source = sent.text.rstrip().lstrip()
                if is_chapter_title(source):
                    # this is likely a chapter title
                    print(f"CHAPTER {chapter_nr} = |{source}|")
                    start_token = "<SOC> "
                    chapter_nr += 1
                    continue 
                canonical = environment.grammar.convert_to_canonical(source)
                natural = environment.grammar.convert_from_canonical(canonical)
                if environment.configuration.no_roundtrip or line.lower().startswith(natural.lower()):
                    sentences.append(canonical)
                else:
                    print(f"ROUNDTRIP {sent} -> {natural} <- {canonical}")
                        
        self.add_to_curriculum(start_token, sentences, environment)
        
        print(f"curriculum sentences = {len(self.sentences)}")

    def write_curriculum(self, filename: str):
        with open(filename, "w", encoding="utf-8-sig") as file:
            for sentence in self.sentences:
                file.write(sentence.get_canonical() + "\n")

    def write_curriculum_natural(self, filename: str, ):
        with open(filename, "w", encoding="utf-8-sig") as file:
            for sentence in story.sentences:
                file.write(sentence.natural + "\n")

    def read_prepocessed(self, filename:str, environment: WriterEnvironment):
        with open(filename, "r", encoding="utf-8-sig") as f:
            lines = f.read().splitlines()
            
            for line in lines:
                if len(line) >= environment.configuration.min_sentence_length:
                    natural = environment.grammar.convert_from_canonical(line)
                    tokens = [self.token_dictionary.add_and_get(token) for token in line.split(" ") if token != ""]
                    self.sentences.append(CurriculumSentence(tokens, natural))
                    if len(self.sentences) >= environment.configuration.max_sentences:
                        break                    
    
            print(f"curriculum sentences = {len(self.sentences)}")

def is_chapter_title(s: str) -> bool:
    if s.lower().startswith("chapter"):
        return True
    return len(s) > 2 and any(c.isupper() for c in s) and not any(c.islower() for c in s)

def preprocess_line(line: str) -> str:
    line = re \
        .sub("[\[\]‑\-—_\“\”‘…]", " ", line) \
        .replace("!)", ",") \
        .replace("),", ",") \
        .replace("[;:()]", ",") \
        .replace("’", "'") \
        .replace("!!!", "!") \
        .replace("...", ".") \
        .replace("Dr.", "Dr") \
        .replace("Mr.", "Mr")
    return line
