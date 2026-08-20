# tokens.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from uuid import UUID
import math
import random

# All possible grammar tokens
GRAMMAR_TOKENS = [
    "<VERB-PRESENT>",
    "<VERB-PRESENT-3S>",
    "<VERB-PRESENT-1S>",
    "<VERB-PAST>",
    "<VERB-ING>",
    "<VERB-INGV>",
    "<VERB-PERFECT>",
    "<NOUN>",
    "<NOUN-PLURAL>",
    "<PRON>",
    "<PRONA>",
    "<DET>",
    "<ADJ>",
    "<ADV>",
    "<ADP>",
    "<PART>",
    "<SCONJ>",
    "<CCONJ>",
    "<NUM>",
    "<PROPN>",
    "<X>",
    "<INTJ>",
    "<PERIOD>",
    "<COMMA>",
    "<EXCLAMATION>",
    "<QUESTION>",
    "<EOL>",
]

TERMINAL_TOKENS = {
    "<PERIOD>",
    "<COMMA>",
    "<EXCLAMATION>",
    "<QUESTION>",
    "<EOL>",
}

END_PUNCTIATION_TOKENS = {
    "<PERIOD>",
    "<EXCLAMATION>",
    "<QUESTION>",
}

ALL_PUNCTIATION_TOKENS = {
    "<PERIOD>",
    "<EXCLAMATION>",
    "<QUESTION>",
    "<COMMA>"
}

# ============================================================
# Token (symbolic only)
# ============================================================

class Token:
    """
    Pure symbolic token:
    - text
    - deterministic ID
    - grammar_index (if grammar token)
    - lemma_index (if lemma token)
    """

    EOL = None
    NONE = None

    def __init__(self, text: str):
        self._text = text
        self._id = self._compute_id(text)
        self._grammar_index: Optional[int] = None
        self._lemma_index: Optional[int] = None

    @staticmethod
    def _compute_id(text: str) -> int:
        id_ = 0
        for ch in text:
            id_ = (id_ + ord(ch)) * 997
        return id_

    @staticmethod
    def _compute_embedding(id_: int, length: int) -> List[float]:
        rng = random.Random((id_ * id_) * 321)
        emb = [rng.gauss(0.0, 1.0) for _ in range(length)]
        norm = math.sqrt(sum(x * x for x in emb))
        if norm > 0:
            emb = [x / norm for x in emb]
        return emb

    # ------------------------------------------------------------
    # Token type checks
    # ------------------------------------------------------------
    def is_eol(self) -> bool:
        return self._text == "<EOL>"

    def is_terminal(self) -> bool:
        return self._text in TERMINAL_TOKENS

    def is_end_punctuation(self) -> bool:
        return self._text in END_PUNCTIATION_TOKENS

    def is_all_punctuation(self) -> bool:
        return self._text in ALL_PUNCTIATION_TOKENS

    def is_grammar(self) -> bool:
        return self._text[0] == '<'

    def is_lemma(self) -> bool:
        return self._text[0] != '<'

    # ------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------
    @property
    def text(self) -> str:
        return self._text

    @property
    def lower_text(self) -> str:
        return self._text.lower()

    @property
    def id(self) -> int:
        return self._id

    @property
    def grammar_index(self) -> Optional[int]:
        return self._grammar_index

    @grammar_index.setter
    def grammar_index(self, idx: int):
        self._grammar_index = idx

    @property
    def lemma_index(self) -> Optional[int]:
        return self._lemma_index

    @lemma_index.setter
    def lemma_index(self, idx: int):
        self._lemma_index = idx

    def __repr__(self):
        return self._text

    def __hash__(self):
        return hash(self._text)

    def __eq__(self, other):
        return isinstance(other, Token) and self._text == other._text


# Initialize static tokens
Token.EOL = Token("<EOL>")
Token.PERIOD = Token("<PERIOD>")
Token.EXCLAMATION = Token("<EXCLAMATION>")
Token.QUESTION = Token("<QUESTION>")
Token.COMMA = Token("<COMMA>")
Token.NONE = Token("<NONE>")

# ============================================================
# TokenLogit
# ============================================================

@dataclass(frozen=True)
class TokenLogit:
    grammar: Token
    lemma: Token
    logit: float

# ============================================================
# TargetToken
# ============================================================

@dataclass(frozen=True)
class TargetToken:
    grammar: Token
    lemma: Token

# Terminal target tokens
TargetToken.TERMINALS = [
    TargetToken(Token.PERIOD, Token.PERIOD),
    TargetToken(Token.EXCLAMATION, Token.EXCLAMATION),
    TargetToken(Token.QUESTION, Token.QUESTION),
    TargetToken(Token.COMMA, Token.COMMA),
    TargetToken(Token.EOL, Token.EOL),    
]

# ============================================================
# TokenPage
# ============================================================

@dataclass
class TokenPage:
    uuid: UUID
    input_tokens: set
    output_tokens: List[TargetToken]
    output_token_set: set = field(init=False)

    def __post_init__(self):
        # Set for fast membership checks
        self.output_token_set = set(self.output_tokens)

    def input_size(self) -> int:
        return len(self.input_tokens)

    def output_size(self) -> int:
        return len(self.output_tokens)

    def add_input_token(self, token: str):
        self.input_tokens.add(token)

    def has_input_token(self, token: str) -> bool:
        return token in self.input_tokens

    def add_output_token(self, token: TargetToken):
        if token not in self.output_token_set:
            self.output_tokens.append(token)
            self.output_token_set.add(token)

    def has_output_token(self, token: TargetToken) -> bool:
        return token in self.output_token_set

    def get_size_text(self) -> str:
        return f"{len(self.input_tokens)} -> {len(self.output_tokens)}"

# ============================================================
# TokenPages
# ============================================================

@dataclass
class TokenPages:
    pages: List[TokenPage] = field(default_factory=list)
    input_cache: Dict[str, TokenPage] = field(default_factory=dict)
    output_cache: Dict[TargetToken, TokenPage] = field(default_factory=dict)

    def __repr__(self):
        return repr(self.pages)

    def add_token_page(self, page: TokenPage):
        self.pages.append(page)

    def get_input_token_page(self, token: str) -> Optional[TokenPage]:
        if token in self.input_cache:
            return self.input_cache[token]
        for page in self.pages:
            if page.has_input_token(token):
                self.input_cache[token] = page
                return page
        return None

    def get_output_token_page(self, token: TargetToken) -> Optional[TokenPage]:
        if token in self.output_cache:
            return self.output_cache[token]
        for page in self.pages:
            if page.has_output_token(token):
                self.output_cache[token] = page
                return page
        return None

# ============================================================
# TokenDictionary
# ============================================================

class TokenDictionary:
    def __init__(self):
        self.map: Dict[str, Token] = {}
        self._next_lemma_index = 0

        # 1. Preload all grammar tokens with fixed indices
        for idx, text in enumerate(GRAMMAR_TOKENS):
            tok = Token(text)
            tok.grammar_index = idx
            self.map[text] = tok

        # 2. Also preload EOL and NONE if needed
        Token.EOL.grammar_index = GRAMMAR_TOKENS.index("<EOL>")
        self.map["<EOL>"] = Token.EOL

        Token.NONE.grammar_index = None
        self.map["<NONE>"] = Token.NONE

    def add_and_get(self, text: str) -> Token:
        # Already known?
        if text in self.map:
            return self.map[text]

        # Create new token
        tok = Token(text)
        self.map[text] = tok

        # Grammar token?
        if tok.is_grammar():
            # Grammar tokens should already be preloaded
            # but if new grammar tokens appear, assign index dynamically
            if tok.text in GRAMMAR_TOKENS:
                tok.grammar_index = GRAMMAR_TOKENS.index(tok.text)
            else:
                raise ValueError(f"Unknown grammar token: {tok.text}")
            return tok

        # Lemma token → assign lemma_index
        tok.lemma_index = self._next_lemma_index
        self._next_lemma_index += 1

        return tok

    def __contains__(self, text: str) -> bool:
        return text in self.map

    def __getitem__(self, text: str) -> Token:
        return self.map[text]

    def __repr__(self):
        return f"TokenDictionary({list(self.map.keys())})"
    
# ============================================================
# LearnedTokenEmbedding
# ============================================================

@dataclass
class LearnedTokenEmbedding:
    embedding: List[float]
    dimension: int
    
    def update(self, target: List[float], alpha: float):
        if len(target) != self.dimension:
            raise Exception(f"Incompatible dimension! {dimension} != {len(target)}")
        orig_alpha = 1.0 - alpha
        for i in range(0, self.dimension):
            self.embedding[i] = self.embedding[i] * orig_alpha + target[i] * alpha
    
    @staticmethod
    def create(token: Token, dimension: int) -> "LearnedTokenEmbedding":
        embedding = Token._compute_embedding(token._id, dimension)
        return LearnedTokenEmbedding(embedding, dimension)

    @staticmethod
    def create_from_id(token_id: int, dimension: int) -> "LearnedTokenEmbedding":
        embedding = Token._compute_embedding(token_id, dimension)
        return LearnedTokenEmbedding(embedding, dimension)

# ============================================================
# LemmaEmbeddingDictionary
# ============================================================

class LemmaEmbeddingDictionary:
    def __init__(self, input_dimension: int, output_dimension: int):
        self.input_embeddings: Dict[int, LearnedTokenEmbedding] = {}
        self.output_embeddings: Dict[int, LearnedTokenEmbedding] = {}
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

    def get_input_embedding(self, lemma: Token) -> LearnedTokenEmbedding:
        embedding = self.input_embeddings.get(lemma._id)
        if embedding:
            return embedding
        embedding = LearnedTokenEmbedding.create(lemma, self.input_dimension)
        self.input_embeddings[lemma._id] = embedding
        return embedding

    def get_output_embedding(self, target: TargetToken) -> LearnedTokenEmbedding:
        key = (target.grammar.id * 100_000_000) + target.lemma.id
        embedding = self.output_embeddings.get(key)
        if embedding:
            return embedding
        embedding = LearnedTokenEmbedding.create_from_id(key, self.output_dimension)
        self.output_embeddings[key] = embedding
        return embedding
