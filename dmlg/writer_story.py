# writer_story.py
from dataclasses import dataclass, field
from typing import List, Set

from .tokens import Token

@dataclass
class WriterSentence:
    tokens: List[Token]
    natural: str
    fixed: str = field(init=False)
    
    def __post_init__(self):
        self.fixed = self.natural

@dataclass
class WriterStory:
    sentences: List[WriterSentence]

    def get_story(self) -> str:
        joined = " ".join([sentence.fixed for sentence in self.sentences])
        return joined
    
    def to_natural(self) -> List[str]:
        result = []
        indent = True
        for sentence in self.sentences:
            line = sentence.fixed
            if line[0] == '$' or line[0] == '#':
                line = line[2:]
                indent = True
            if indent:
                result.append("\t" + line)
                indent = False
            else:
                result.append(line)
        return result

# ------------------------------------------------------------
# Generation parameters
# ------------------------------------------------------------

@dataclass
class WriterParams:
    amount: int = 1 # number of books
    lines: int = 1000 # lines per book
    min_lines_chapter: int = 20 # minimum lines per chapter
    min_lines_paragraph: int = 3 # minimum lines per paragraph
    min_words: int = 8 # minimum words per sentence
    max_words: int = 30 # maximum words per sentence
    max_chapters: int = 1000 # maximum number of chapters
    from_line_fraction: float = 0.0 # start line fraction
    to_line_fraction: float = 1.0 # end line fraction
    prompt: List[str] = None # prompt lines
    keywords: Set[str] = None # keywords for beam-search
    lemma_blacklist: Set[str] = None # blacklisted lemma set
    beam_search: bool = False # use beam-search
    max_tokens: int = 70 # maximum grammar + lemma tokens per sentence
    max_attempts: int = 20000 # number of retry attempts to generate sentence 
    top_k: int = 50 # amount of eligible tokens with highest score for sampling
    temperature: float = 0.001 # small scoring bias to equalize token scoring
    nr_of_beams: int = 3 # beams to use during beam-search
    beam_alpha: float = 0.8 # beam score damping
    beam_jitter: float = 0.5 # random jitter to add to beam scores
    beam_attempts: int = 3 # attempts to try creating a valid sentence with beam-search
    beam_temperature: float = 0.8 # temperature for token softmax scoring
    top_boost = [1.3, 1.2, 1.1] # score boost for top tokens
    enable_paging = True # use graph or pure MLP mode
    max_pageless_vocab = 2048 # maximum random subset of vocab to use for MLP mode
