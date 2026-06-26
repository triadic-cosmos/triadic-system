# tokens.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from uuid import UUID
import math
import random

# Grammar tokens without lemma token
TERMINAL_TOKENS = {
    "<PERIOD>",
    "<COMMA>",
    "<EXCLAMATION>",
    "<QUESTION>",
    "<EOL>",
}

# Punctuation tokens to end a sentence
PUNCTIATION_TOKENS = {
    "<PERIOD>",
    "<EXCLAMATION>",
    "<QUESTION>",    
}

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

    def is_terminal(self) -> bool:
        return self._text in TERMINAL_TOKENS

    def is_punctuation(self) -> bool:
        return self._text in PUNCTIATION_TOKENS

    def is_grammar(self) -> bool:
        return self._text[0] == '<'

    def is_lemma(self) -> bool:
        return self._text[0] != '<'

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
# TokenPage
# ============================================================

# ============================================================
# TokenPage
# ============================================================

@dataclass
class TokenPage:
    uuid: UUID
    input_tokens: set
    output_tokens: List[Token]
    output_token_set: set = field(init=False)
    output_index: Dict[Token, int] = field(init=False)

    def __post_init__(self):
        # Set for fast membership checks
        self.output_token_set = set(self.output_tokens)

        # Dict for fast index lookup
        self.output_index = {tok: idx for idx, tok in enumerate(self.output_tokens)}

    def input_size(self) -> int:
        return len(self.input_tokens)

    def output_size(self) -> int:
        return len(self.output_tokens)

    def add_input_token(self, token: str):
        self.input_tokens.add(token)

    def has_input_token(self, token: str) -> bool:
        return token in self.input_tokens

    def add_output_token(self, token: Token):
        if token not in self.output_token_set:
            self.output_tokens.append(token)
            self.output_token_set.add(token)
            # update index map
            self.output_index[token] = len(self.output_tokens) - 1

    def has_output_token(self, token: Token) -> bool:
        return token in self.output_token_set

    def get_output_index(self, token: Token) -> int:
        """
        Returns the index of the token in the output list.
        Raises KeyError if token not present.
        """
        return self.output_index[token]

    def get_first_output_token(self) -> Token:
        return self.output_tokens[0]

    def get_size_text(self) -> str:
        return f"{len(self.input_tokens)} -> {len(self.output_tokens)}"
# ============================================================
# TokenPage
# ============================================================

@dataclass
class TokenPage:
    uuid: UUID
    input_tokens: set
    output_tokens: List[Token]
    output_token_set: set = field(init=False)
    output_index: Dict[Token, int] = field(init=False)

    def __post_init__(self):
        # Set for fast membership checks
        self.output_token_set = set(self.output_tokens)

        # Dict for fast index lookup
        self.output_index = {tok: idx for idx, tok in enumerate(self.output_tokens)}

    def input_size(self) -> int:
        return len(self.input_tokens)

    def output_size(self) -> int:
        return len(self.output_tokens)

    def add_input_token(self, token: str):
        self.input_tokens.add(token)

    def has_input_token(self, token: str) -> bool:
        return token in self.input_tokens

    def add_output_token(self, token: Token):
        if token not in self.output_token_set:
            self.output_tokens.append(token)
            self.output_token_set.add(token)
            # update index map
            self.output_index[token] = len(self.output_tokens) - 1

    def has_output_token(self, token: Token) -> bool:
        return token in self.output_token_set

    def get_output_index(self, token: Token) -> int:
        """
        Returns the index of the token in the output list.
        Raises KeyError if token not present.
        """
        return self.output_index[token]

    def get_first_output_token(self) -> Token:
        return self.output_tokens[0]

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

# ============================================================
# TokenCodeBook
# ============================================================

class TokenCodeBook:
    def __init__(self, output_dim: int = 32, max_tokens: int = 2000):
        self.output_dim = output_dim
        self.max_tokens = max_tokens

        self.one_hot_dim = output_dim // 4              # 8
        self.combo_dim = output_dim - self.one_hot_dim  # 24

        # one-hot bits: 0..7
        # tri-hot bits: 8..31

        self._triple_codes = []
        for i in range(self.combo_dim):
            for j in range(i + 1, self.combo_dim):
                for k in range(j + 1, self.combo_dim):
                    self._triple_codes.append((i, j, k))

        needed = max(0, max_tokens - self.one_hot_dim)
        if needed > len(self._triple_codes):
            raise ValueError(
                f"Not enough triple-hot codes: need {needed}, have {len(self._triple_codes)}"
            )

    def get_bits(self, index: int):
        if index < self.one_hot_dim:
            # pure one-hot in first 8 bits
            return [index]
        else:
            triple_index = index - self.one_hot_dim
            i, j, k = self._triple_codes[triple_index]
            # shift to combo-range: 8..31
            return [self.one_hot_dim + i,
                    self.one_hot_dim + j,
                    self.one_hot_dim + k]

    def get_vector(self, index: int):
        vec = [0.0] * self.output_dim
        for pos in self.get_bits(index):
            vec[pos] = 1.0
        return vec
