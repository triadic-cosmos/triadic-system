# context.py
from dataclasses import dataclass, field
from typing import List

from .tokens import Token, TargetToken, HISTORY_TOKENS, TRACKER_TOKENS
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

    def clear(self):
        emb = self.lemma_embedding_dict.get_embedding(TargetToken.EOL).embedding
        self.tokens = emb * self.amount
        self.position = 0
    
    def add(self, token: TargetToken):
        emb = self.lemma_embedding_dict.get_embedding(token).embedding
        self.tokens = emb + self.tokens[:-len(emb)]
        self.position += 1
    
    def get(self):
        cur_pos = self.position / self.divider
        return self.tokens + [cur_pos]
    
    def copy(self) -> "CurrentSentence":
        copy = CurrentSentence(self.lemma_embedding_dict, self.divider, self.amount)
        copy.tokens = self.tokens.copy()
        copy.position = self.position
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
     
    def copy(self) -> "EntityTracker":
        copy = EntityTracker(self.tracking, self.alpha)
        copy.state = self.state.copy()
        return copy
     
# ============================================================
# Context Window
# ============================================================

@dataclass
class ContextWindow:
    configuration: Configuration
    lemma_embedding_dict: any
    tracking: any
        
    def __post_init__(self):
        self._cas_vector = []

        self._token_histories = []
        self._token_histories_start = [] 
        for i in range(len(self.configuration.history_size)):
            dimension = self.configuration.history_size[i]
            alpha = self.configuration.history_alpha[i]
            self._token_histories.append(TokenHistory(dimension, alpha))
            self._token_histories_start.append(TokenHistory(dimension, alpha))

        self._entity_tracker = EntityTracker(self.tracking, self.configuration.tracker_alpha)

        self._sentence_memory = [0] * self.configuration.narrative_sentences \
                                    * self.configuration.token_embedding_dimension
        
        self._current_tokens = []
        self._current_sentence = CurrentSentence(
            self.lemma_embedding_dict,
            self.configuration.position_divider,
            self.configuration.last_embeddings)

        self.clear_current_sentence()

    def clear_current_sentence(self):
        for i in range(len(self._token_histories)):
            self._token_histories[i] = self._token_histories_start[i].copy()

        self._current_tokens = []
        self._current_sentence.clear()
        
    def start_sentence(self):
        for i in range(len(self._token_histories)):
            self._token_histories_start[i] = self._token_histories[i].copy()
        
        self.clear_current_sentence()

    def last_token(self) -> Token:
        if self._current_tokens:
            return self._current_tokens[-1].lemma
        return Token.EOL

    def forelast_token(self) -> Token:
        if self._current_tokens:
            last: TargetToken = self._current_tokens[-1]
            if last.lemma.is_lemma():
                return last.grammar
            if len(self._current_tokens) == 1:
                return Token.EOL
            forelast: TargetToken = self._current_tokens[-2]
            return forelast.lemma
        return Token.EOL
    
    def add_target_token(self, token: TargetToken):
        self._current_tokens.append(token)
        self._current_sentence.add(token)
        
        if token.lemma.is_lemma() and \
            token.grammar.text in HISTORY_TOKENS and \
            token.lemma.text not in HISTORY_BLACKLIST:
            emb = self.lemma_embedding_dict.get_embedding(token).embedding
            for token_history in self._token_histories:
                token_history.add(emb)
            
    def update_sentence_memory(self, vector: List[float]):
        self._sentence_memory = vector + self._sentence_memory[:-len(vector)]
        
    def update_entity_tracking(self, tokens: List[Token]):
        self._entity_tracker.update_from_sentence(tokens)
    
    def copy(self) -> "ContextWindow":
        ctx: ContextWindow = ContextWindow(self.configuration, self.lemma_embedding_dict, self.tracking)

        for i in range(len(self._token_histories)):
            ctx._token_histories[i] = self._token_histories[i].copy()
            ctx._token_histories_start[i] = self._token_histories_start[i]                                                            

        ctx._current_tokens = self._current_tokens.copy()        
        ctx._current_sentence = self._current_sentence.copy()

        ctx._cas_vector = self._cas_vector
        ctx._entity_tracker = self._entity_tracker.copy()
        ctx._sentence_memory = self._sentence_memory.copy()
        return ctx
        
    def set_cas_vector(self, cas_vector: List[float]):
        self._cas_vector = cas_vector
    
    @property
    def position(self) -> int:
        return self._current_sentence.position
        
    @property
    def cas_vector(self) -> List[float]:
        return self._cas_vector
    
    @property
    def current_tokens(self) -> List[TargetToken]:
        return self._current_tokens
            
    def get_current_embedding(self) -> List[float]:
        return self._current_sentence.get()
        
    def get_sentence_memory_embedding(self) -> List[float]:
        return self._sentence_memory
       
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
    
    def copy(self) -> "ModelInput":
        return ModelInput(self.window.copy(), self.line_position, True, True)

# ============================================================
# InputEncoder
# ============================================================

@dataclass(frozen=True)
class InputEncoder:
    def encode_sentence(self, model_input: ModelInput) -> List[float]:
        sentence_embedding = model_input.window.get_sentence_memory_embedding()
        entity_embedding = model_input.window.get_entity_tracker_embedding()
        line_position = model_input.line_position
        cas_vector = model_input.window.cas_vector
        
        return (
            sentence_embedding
            + entity_embedding
            + line_position
            + cas_vector
        )        
        
    def encode_token(self, model_input: ModelInput) -> List[float]:
        current_embedding = model_input.window.get_current_embedding()
        sentence_embedding = model_input.window.get_sentence_memory_embedding()
        entity_embedding = model_input.window.get_entity_tracker_embedding()
        token_history = model_input.window.get_token_history_embedding()
        line_position = model_input.line_position
        cas_vector = model_input.window.cas_vector

        return (
            current_embedding
            + sentence_embedding
            + entity_embedding
            + token_history
            + line_position
            + cas_vector
        )
