# entity_tracker.py
from dataclasses import dataclass, field

from .config import Configuration
from .curriculum import Curriculum
from .tokens import Token, TRACKER_TOKENS

@dataclass
class TrackerEntity:
    lemma: str
    grammar: str
    
    def __post_init__(self):
        self.min_fraction = 1.0
        self.max_fraction = 0.0
        self.amount = 0
        
    def score(self):
        if not self.lemma[0].isalpha():
            return 0
        score = self.amount + 2 * (self.max_fraction - self.min_fraction)
        if self.grammar == "<PROPN>":
            score += 3
        return score

    def __str__(self):
        return f"{self.lemma} {self.grammar} {self.amount} {self.score()}"

@dataclass
class TrackerMap:
    config: Configuration
    entities: dict = field(default_factory=dict)
    partitions: dict = field(default_factory=dict)

    def add_entities(self, curriculum: Curriculum):
        line_nr: int = 0
        last_line = len(curriculum.sentences) - 1
        
        for sentence in curriculum.sentences:
            line_fraction = line_nr / last_line
            line_nr += 1
            grammar = None
            
            for token in sentence.tokens:
                if token.text in TRACKER_TOKENS:
                    grammar = token.text
                elif grammar:
                    if token.is_lemma():
                        if token.text in self.entities:        
                            entity = self.entities[token.text]
                        else:
                            entity = TrackerEntity(token.text, grammar)
                            self.entities[token.text] = entity
                        entity.min_fraction = min(entity.min_fraction, line_fraction)
                        entity.max_fraction = max(entity.max_fraction, line_fraction)
                        entity.amount += 1
                    grammar = None

        print(f"entities = {len(self.entities)}")

    def partition_entities(self):
        keys = sorted(self.entities, key=lambda k: self.entities[k].score(), reverse=True)
        partition_keys = keys[:self.config.tracker_entities]
        index = 0
        
        for k in partition_keys:
            self.partitions[k] = index
            index += 1
            if index <= 20:
                print(f"[{index}] {self.entities[k]}")            
        
        print(f"partitions = {len(self.partitions)}")
