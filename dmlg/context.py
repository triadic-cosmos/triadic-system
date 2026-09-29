# context.py
from dataclasses import dataclass, field
from typing import List

from .tokens import Token, HISTORY_TOKENS, TRACKER_TOKENS
from .narrative_memory import NarrativeMemory
from .config import Configuration

HISTORY_BLACKLIST = {
    # Copulas 
    "be",
    # Auxiliaries
    "have",
    "do",
    # Modals 
    "will",
    "can",
    "shall",
    "may",
    "must",
    "might",
    "could",
    "would",    
    "should"
}

# ============================================================
# Current Sentence
# ============================================================

@dataclass
class CurrentSentence:
    lemma_embedding_dict: any
    divider: float    
    amount: int
    size: int

    def clear(self):
        emb = self.lemma_embedding_dict.get_input_embedding(Token.EOL).embedding[:self.size]
        self.grammar = emb * self.amount
        self.lemma = emb * self.amount
        self.position = 0
        self.punctuation = 0.0
        
    def add(self, token: Token):
        emb = self.lemma_embedding_dict.get_input_embedding(token).embedding[:self.size]        
        if token.is_lemma():
            self.position += 1
            self.punctuation = 0.0
            self.lemma = emb + self.lemma[: len(self.lemma) - self.size]
        elif token.is_terminal():
            self.position += 1
            self.punctuation = 1.0
        else:
            self.punctuation = 0.0
            self.grammar = emb + self.grammar[: len(self.grammar) - self.size]
  
    def get(self):
        cur_pos = self.position / self.divider
        return self.grammar + self.lemma + [self.punctuation] + [cur_pos]
    
    def copy(self) -> "CurrentSentence":
        copy = CurrentSentence(self.lemma_embedding_dict, self.amount, self.size, self.divider)
        copy.grammar = self.grammar.copy()
        copy.lemma = self.lemma.copy()
        copy.position = self.position
        copy.punctuation = self.punctuation
        return copy

# ============================================================
# Token History
# ============================================================
@dataclass
class TokenHistory:
    dimension: int
    alpha: float

    def __post_init__(self):
        self.history = [0] * self.dimension
        self.alpha2 = 1.0 - self.alpha
    
    def add(self, embedding: List[float]):
        self.history = self.history[1:] + self.history[:1]
        for i in range(self.dimension):
            self.history[i] = self.history[i] * self.alpha + embedding[i] * self.alpha2
        
    def get(self):
        return self.history
        
    def copy(self) -> "TokenHistory":
        copy = TokenHistory(self.dimension, self.alpha)
        copy.history = self.history.copy()
        return copy

# ============================================================
# Entity Tracker
# ============================================================

@dataclass
class EntityTracker:
    tracking: dict
    alpha: float
 
    def __post_init__(self):
        self.state = [0] * len(self.tracking)

    def get_state(self) -> List[float]:
        return self.state

    def update_from_sentence(self, tokens: List[Token]):
        grammar = False
        self.state = [t * self.alpha for t in self.state]
        
        for token in tokens:
            if token.text in TRACKER_TOKENS:
                grammar = True
            elif grammar:
                grammar = False
                if token.text in self.tracking:
                    self.state[self.tracking[token.text]] = 1.0
                    
# ============================================================
# Context Window
# ============================================================

@dataclass
class ContextWindow:
    configuration: Configuration
    lemma_embedding_dict: any
    tracking: any
    
    def __post_init__(self):
        self._token_histories = []
        self._token_histories_start = [] 
        for i in range(len(self.configuration.history_size)):
            dimension = self.configuration.history_size[i]
            alpha = self.configuration.history_alpha[i]
            self._token_histories.append(TokenHistory(dimension, alpha))
            self._token_histories_start.append(TokenHistory(dimension, alpha))

        self._narrative_memory = NarrativeMemory(self.lemma_embedding_dict, self.configuration)
        self._narrative_memory_embedding = None

        self._entity_tracker = EntityTracker(self.tracking, self.configuration.tracker_alpha)

        self._current_tokens = []
        self._current_sentence = CurrentSentence(
            self.lemma_embedding_dict,
            self.configuration.position_divider,
            self.configuration.last_embeddings,
            self.configuration.last_embedding_size)
        self.clear_current_sentence()

    def clear_current_sentence(self):
        for i in range(len(self._token_histories)):
            self._token_histories[i] = self._token_histories_start[i].copy()

        self._current_tokens = []
        self._current_sentence.clear()

        self._last_token = Token.EOL
        self._forelast_token = Token.EOL
        self._last_lemma = Token.EOL

    def start_sentence(self):
        for i in range(len(self._token_histories)):
            self._token_histories_start[i] = self._token_histories[i].copy()
        
        self.clear_current_sentence()

    def last_token(self) -> Token:
        return self._last_token

    def forelast_token(self) -> Token:
        return self._forelast_token

    def last_grammar_token(self) -> Token:
        if self._last_token.is_grammar():
            return self._last_token
        return self._forelast_token

    def last_lemma_token(self) -> str:
        if self._last_token.is_eol():
            return Token.EOL.text
        return self._last_lemma.text

    def add_token(self, token: Token):
        self._current_tokens.append(token)
        self._current_sentence.add(token)
        self._forelast_token = self._last_token
        self._last_token = token

        if token.is_eol():
            self._last_lemma = token

        elif token.is_lemma():
            self._last_lemma = token
            if self._forelast_token.text in HISTORY_TOKENS and \
                token.text not in HISTORY_BLACKLIST:
                    emb = self.lemma_embedding_dict.get_input_embedding(token).embedding
                    for token_history in self._token_histories:
                        token_history.add(emb)
                                
    def update_narrative_memory(self, tokens: List[Token]):
        self._narrative_memory.update_from_sentence(tokens)
        self._narrative_memory_embedding = None
    
    def update_entity_tracking(self, tokens: List[Token]):
        self._entity_tracker.update_from_sentence(tokens)
    
    def copy_current(self) -> "ContextWindow":
        ctx: ContextWindow = ContextWindow(self.configuration, self.lemma_embedding_dict)

        for i in range(len(self._token_histories)):
            ctx._token_histories[i] = self._token_histories[i].copy()
            ctx._token_histories_start[i] = self._token_histories_start[i]                                                            

        ctx._current_tokens = self._current_tokens.copy()        
        ctx._current_sentence = self._current_sentence.copy()
        ctx._last_token = self._last_token
        ctx._forelast_token = self._forelast_token
        ctx._last_lemma = self._last_lemma
        return ctx
    
    @property
    def current_tokens(self) -> List[Token]:
        return self._current_tokens
            
    def get_current_embedding(self) -> List[float]:        
        return self._current_sentence.get()

    def get_narrative_memory_embedding(self) -> List[float]:
        if self._narrative_memory_embedding == None:
            self._narrative_memory_embedding = self._narrative_memory.get_state()
        return self._narrative_memory_embedding
    
    def get_entity_tracker_embedding(self) -> List[float]:
        return self._entity_tracker.get_state()
    
    def get_token_history_embedding(self) -> List[float]:
        embedding = []
        for token_history in self._token_histories:
            embedding += token_history.get() 
        return embedding
    
# ============================================================
# ModelInput
# ============================================================

@dataclass(frozen=True)
class ModelInput:
    window: ContextWindow
    line_position: List[float]
    allow_chapter: bool
    allow_paragraph: bool

# ============================================================
# InputEncoder
# ============================================================

@dataclass(frozen=True)
class InputEncoder:
    def encode(self, model_input: ModelInput) -> List[float]:
        current_embedding = model_input.window.get_current_embedding()
        narrative_embedding = model_input.window.get_narrative_memory_embedding()
        entity_embedding = model_input.window.get_entity_tracker_embedding()
        token_history = model_input.window.get_token_history_embedding()
        line_position = model_input.line_position

        return (
            current_embedding
            + narrative_embedding
            + entity_embedding
            + token_history
            + line_position
        )
