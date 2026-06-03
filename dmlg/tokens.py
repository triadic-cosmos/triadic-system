# tokens.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from uuid import UUID
import math
import random


# ============================================================
# Token
# ============================================================

MEMORY_EMBEDDING_SIZE = 32
LARGE_EMBEDDING_SIZE = 4
MEDIUM_EMBEDDING_SIZE = 2


class Token:
    """
    Deterministic token with:
    - text
    - deterministic ID
    - deterministic large embedding
    - deterministic medium embedding
    """

    # initialized after class definition
    EOL = None  
    NONE = None

    def __init__(self, text: str):
        self._text = text
        self._id = self._compute_id(text)
        self._memory_embedding = self._compute_embedding(self._id, MEMORY_EMBEDDING_SIZE)
        self._large_embedding = self._compute_embedding(self._id, LARGE_EMBEDDING_SIZE)
        self._medium_embedding = self._compute_embedding(self._id, MEDIUM_EMBEDDING_SIZE)
        self._small_embedding = self._compute_small_embedding(self._id)

    def is_eol(self) -> bool:
        return self._text == "<EOL>"

    def is_lemma(self) -> bool:
        if self._text[0] == '<':
            return False
        return True

    @property
    def text(self) -> str:
        return self._text

    @property
    def id(self) -> int:
        return self._id

    @property
    def memory_embedding(self) -> List[float]:
        return self._memory_embedding

    @property
    def large_embedding(self) -> List[float]:
        return self._large_embedding

    @property
    def medium_embedding(self) -> List[float]:
        return self._medium_embedding

    @property
    def small_embedding(self) -> List[float]:
        return self._small_embedding

    @staticmethod
    def _compute_id(text: str) -> int:
        id_ = 0
        for ch in text:
            id_ = (id_ + ord(ch)) * 997
        return id_

    @staticmethod
    def _compute_small_embedding(id_: int) -> float:
        rng = random.Random((id_ * id_) * 123)
        return rng.gauss(0.0, 1.0)

    @staticmethod
    def _compute_embedding(id_: int, length: int) -> List[float]:
        rng = random.Random((id_ * id_) * 321)
        emb = [rng.gauss(0.0, 1.0) for _ in range(length)]
        norm = math.sqrt(sum(x * x for x in emb))
        if norm > 0:
            emb = [x / norm for x in emb]
        return emb

    def __repr__(self):
        return self._text

    def __hash__(self):
        return hash(self._text)

    def __eq__(self, other):
        return isinstance(other, Token) and self._text == other._text


# Initialize static EOL token
Token.EOL = Token("<EOL>")
Token.NONE = Token("<NONE>")


# ============================================================
# TokenLogit
# ============================================================

@dataclass(frozen=True)
class TokenLogit:
    token: Token
    logit: float


# ============================================================
# TokenMapping
# ============================================================

@dataclass
class TokenMapping:
    token_to_position: Dict[Token, int]
    position_to_token: Dict[int, Token]

    def encode(self, token: Token) -> List[float]:
        """
        One-hot encode a token:
        - vector length = number of output tokens
        - 1.0 at the mapped position
        - 0.0 elsewhere
        """
        size = self.size()
        vec = [0.0] * size
        pos = self.map_to_position(token)
        vec[pos] = 1.0
        return vec

    def map_to_position(self, token: Token) -> int:
        return self.token_to_position[token]

    def map_to_token(self, pos: int) -> Token:
        return self.position_to_token[pos]

    def size(self) -> int:
        return len(self.token_to_position)


# ============================================================
# TokenPage
# ============================================================

@dataclass
class TokenPage:
    uuid: UUID
    input_tokens: set
    output_tokens: List[Token]
    output_token_set: set = field(init=False)

    def __post_init__(self):
        self.output_token_set = set(self.output_tokens)

    def input_size(self) -> int:
        return len(self.input_tokens)

    def output_size(self) -> int:
        return len(self.output_tokens)

    def is_deterministic(self) -> bool:
        return self.output_size() <= 1

    def add_input_token(self, token: str):
        self.input_tokens.add(token)

    def has_input_token(self, token: str) -> bool:
        return token in self.input_tokens

    def add_output_token(self, token: Token):
        if token not in self.output_token_set:
            self.output_tokens.append(token)
            self.output_token_set.add(token)

    def has_output_token(self, token: Token) -> bool:
        return token in self.output_token_set

    def get_first_output_token(self) -> Token:
        return self.output_tokens[0]

    def to_output_mapping(self) -> TokenMapping:
        token_to_pos = {}
        pos_to_token = {}
        for i, tok in enumerate(self.output_tokens):
            token_to_pos[tok] = i
            pos_to_token[i] = tok
        return TokenMapping(token_to_pos, pos_to_token)

    def get_size_text(self) -> str:
        return f"{len(self.input_tokens)} -> {len(self.output_tokens)}"

# ============================================================
# TokenPages
# ============================================================

@dataclass
class TokenPages:
    pages: List[TokenPage] = field(default_factory=list)
    input_cache: Dict[Token, TokenPage] = field(default_factory=dict)
    output_cache: Dict[Token, TokenPage] = field(default_factory=dict)

    def __repr__(self):
        return repr(self.pages)

    def get_first_token_page(self) -> Optional[TokenPage]:
        return self.pages[0] if self.pages else None

    def add_token_page(self, page: TokenPage):
        self.pages.append(page)

    def get_input_token_page(self, token: Token) -> Optional[TokenPage]:
        if token in self.input_cache:
            return self.input_cache[token]
        for page in self.pages:
            if page.has_input_token(token):
                self.input_cache[token] = page
                return page
        return None

    def get_output_token_page(self, token: Token) -> Optional[TokenPage]:
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
    """
    Minimal dictionary that ensures:
    - deterministic token creation
    - no duplicates
    """

    def __init__(self):
        self.map: Dict[str, Token] = {}

    def add_and_get(self, text: str) -> Token:
        if text in self.map:
            return self.map[text]
        tok = Token(text)
        self.map[text] = tok
        return tok

    def __contains__(self, text: str) -> bool:
        return text in self.map

    def __getitem__(self, text: str) -> Token:
        return self.map[text]

    def __repr__(self):
        return f"TokenDictionary({list(self.map.keys())})"
