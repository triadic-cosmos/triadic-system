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

@dataclass
class CurriculumSentence:
    tokens: List[Token]
    natural: str

    def get_canonical(self) -> str:
        return " ".join([token.text for token in self.tokens])

@dataclass
class CurriculumStory:
    sentences: List[CurriculumSentence]

@dataclass(frozen=True)
class Curriculum:
    stories: List[CurriculumStory] = field(default_factory=list)
    token_dictionary: TokenDictionary = field(default_factory=TokenDictionary) 

    def get_random_story(self, rng:random.Random) -> CurriculumStory:
        return self.stories[rng.randrange(0, len(self.stories))]

    # ============================================================
    # Curriculum reader
    # ============================================================

    def add_to_curriculum(self, sentences: List[str], environment: WriterEnvironment) -> CurriculumStory:
        combined: str = " ".join(sentences)
        split = re.split("<SPLIT>", combined)
        
        curriculum_sentences = []
        i = 1
        for s in split:
            if len(s) < environment.configuration.min_sentence_length:
                continue
            if not s.endswith("<PERIOD>") and not s.endswith("<EXCLAMATION>") and not s.endswith("<QUESTION>"):
                continue         
            eol = s.replace("<COMMA> <PERIOD>", "<PERIOD>") + " <EOL>"
            tokens = [self.token_dictionary.add_and_get(t) for t in eol.split(" ") if t != ""]
            natural = environment.grammar.convert_from_canonical(eol)
            curriculum_sentences.append(CurriculumSentence(tokens, natural))
            print(f"{i}. {natural}")
            i += 1

        if len(curriculum_sentences) < environment.configuration.min_story_lines:
            return
        
        story = CurriculumStory(curriculum_sentences)
        self.stories.append(story)
        return story

    def read_curriculum(self, filename: str, environment: WriterEnvironment):
        with open(filename, "r", encoding='utf-8-sig') as f:
            lines = f.read().splitlines()

        sentences = []

        for line in lines:
            line = preprocess_line(line)
            if len(line) < 5:
                self.add_to_curriculum(sentences, environment)
                sentences = []
                if len(self.stories) >= environment.configuration.max_stories:
                    break
                continue
                
            doc = environment.grammar.nlp(line)
            
            for sent in doc.sents:
                source = sent.text
                canonical = environment.grammar.convert_to_canonical(source)
                natural = environment.grammar.convert_from_canonical(canonical)
                if environment.configuration.no_roundtrip or line.lower().startswith(natural.lower()):
                    sentences.append(canonical)
                else:
                    print(f"ROUNDTRIP {sent} -> {natural} <- {canonical}")

                        
        self.add_to_curriculum(sentences, environment)                       
        
        print(f"curriculum stories = {len(self.stories)}")

    def write_curriculum(self, filename: str):
        with open(filename, "w", encoding="utf-8-sig") as file:
            for story in self.stories:
                for sentence in story.sentences:
                    file.write(sentence.get_canonical() + "\n")
                file.write("\n")

    def write_curriculum_natural(self, filename: str, ):
        with open(filename, "w", encoding="utf-8-sig") as file:
            for story in self.stories:
                for sentence in story.sentences:
                    file.write(sentence.natural + "\n")
                file.write("\n")

    def read_prepocessed(self, filename:str, environment: WriterEnvironment):
        with open(filename, "r", encoding="utf-8-sig") as f:
            lines = f.read().splitlines()
            sentences = []
            
            for line in lines:
                if line == "":
                    if len(sentences) > 0:                
                        self.stories.append(CurriculumStory(sentences))
                        sentences = []
                        if len(self.stories) >= environment.configuration.max_stories:
                            break                    
                else:                    
                    natural = environment.grammar.convert_from_canonical(line)
                    tokens = [self.token_dictionary.add_and_get(token) for token in line.split(" ") if token != ""]
                    sentences.append(CurriculumSentence(tokens, natural))
                    
            if (len(sentences) > 0):                
                self.stories.append(CurriculumStory(sentences))
            print(f"curriculum stories = {len(self.stories)}")
                        
def preprocess_line(line: str) -> str:
    line = re \
        .sub("[\[\]‑\-—_\“\”‘]", " ", line) \
        .replace("!)", ",") \
        .replace("),", ",") \
        .replace("[;:()]", ",") \
        .replace("’", "'") \
        .replace("!!!", "!") \
        .replace("...", ".") \
        .replace("Dr.", "Dr") \
        .replace("Mr.", "Mr")
    return line
