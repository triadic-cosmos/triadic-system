# context.py
from dataclasses import dataclass, field
from typing import List

from .tokens import Token
from .sentence_encoder import EncodedSentence
from .narrative_memory import NarrativeMemory
from .config import Configuration

# ============================================================
# Current Sentence
# ============================================================

@dataclass
class CurrentSentence:
    configuration: Configuration
    
    def __post_init__(self):
        self.max_lemmas = self.configuration.content_max_lemmas
        self.cut = self.configuration.sentence_medium_embedding_size
        self.last_position = self.max_lemmas - 1
    
    def clear(self):
        self.position = 0
        self.current_embedding = [
            [0.0] * self.cut for _ in range(self.max_lemmas)
        ]
        
    def add(self, embedding: List[float]):
        if self.position < self.last_position:
            vec = embedding[:self.cut]
            self.current_embedding[self.position] = vec
            self.position += 1
            
    def get(self):
        cur_pos = self.position / self.last_position
        flat = []
        for vec in self.current_embedding:
            flat.extend(vec)
        return [cur_pos] + flat
    
    def copy(self) -> "CurrentSentence":
        copy = CurrentSentence(self.configuration)
        copy.position = self.position
        copy.current_embedding = [vec.copy() for vec in self.current_embedding]
        return copy
      
# ============================================================
# Context Window
# ============================================================
      
@dataclass
class ContextWindow:
    configuration: Configuration
    lemma_embedding_dict: any 

    def __post_init__(self):
        empty_sentence = EncodedSentence.make_empty_sentence(self.configuration)
        self._current_tokens = []
        self._current_grammar_sentence = CurrentSentence(self.configuration)        
        self._current_lemma_sentence = CurrentSentence(self.configuration)
        self.clear_current_sentence()
        self._sentences = [empty_sentence] * self.configuration.context_max_sentences
        self._generator_history_embedding = None
        self._generator_history_sentences = self.configuration.generator_history_sentences
        self._narrative_memory = NarrativeMemory(self.lemma_embedding_dict, self.configuration)
        self._narrative_memory_embedding = None
        
    def clear_current_sentence(self):
        self._current_tokens = []
        self._current_grammar_sentence.clear()
        self._current_lemma_sentence.clear()

        self._last_token = Token.EOL
        self._forelast_token = Token.EOL
        self._last_lemma = Token.EOL

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
        self._forelast_token = self._last_token
        self._last_token = token

        if token.is_eol():
            self._last_lemma = token

        elif token.is_lemma():
            self._last_lemma = token
            emb = self.lemma_embedding_dict.get_input_embedding(token).embedding
            self._current_lemma_sentence.add(emb)
            
        else:
            emb = self.lemma_embedding_dict.get_input_embedding(token).embedding
            self._current_grammar_sentence.add(emb)
            
    def add_sentence(self, sentence: EncodedSentence):
        if len(self._sentences) == self.configuration.context_max_sentences:
            self._sentences.pop()
        self._sentences.insert(0, sentence)
        self._generator_history_embedding = None
        self._evaluator_history_embedding = None
        self.clear_current_sentence()
            
    def update_narrative_memory(self, tokens: List[Token]):
        self._narrative_memory.update_from_sentence(tokens)
        self._narrative_memory_embedding = None
            
    def get_current_embedding(self) -> List[float]:
        if self._forelast_token.is_eol():
            # end of line
            token1 = self._forelast_token
            token2 = self._forelast_token
            comma = [0.0]
        elif self._forelast_token.is_all_punctuation():
            # punctuation
            if len(self._current_tokens) >= 4:
                token1 = self._current_tokens[-4]
                token2 = self._current_tokens[-3]
            else:
                token1 = Token.EOL
                token2 = Token.EOL
                print(f"Unexpected punctuation : {self._current_tokens}")
            comma = [1.0]
        elif self._forelast_token.is_lemma():
            # lemma
            token1 = self._current_tokens[-3]
            token2 = self._forelast_token
            comma = [0.0]
        else:
            # grammar
            token1 = self._forelast_token
            token2 = self._last_token
            comma = [0.0]
                    
        last_size = self.configuration.last_embedding_size            
        embedding1 = self.lemma_embedding_dict.get_input_embedding(token1).embedding[:last_size]
        embedding2 = self.lemma_embedding_dict.get_input_embedding(token2).embedding[:last_size]
        return embedding1 + embedding2 + comma + \
               self._current_grammar_sentence.get() + \
               self._current_lemma_sentence.get()
    
    def copy_current(self) -> "ContextWindow":
        ctx: ContextWindow = ContextWindow(self.configuration, self.lemma_embedding_dict)
        ctx._current_tokens = self._current_tokens.copy()
        ctx._current_grammar_sentence = self._current_grammar_sentence.copy()
        ctx._current_lamma_sentence = self._current_lemma_sentence.copy()
        ctx._last_token = self._last_token
        ctx._forelast_token = self._forelast_token
        ctx._last_lemma = self._last_lemma
        ctx._sentences = self._sentences
        ctx._generator_history_embedding = self._generator_history_embedding
        return ctx
    
    @property
    def current_tokens(self) -> List[Token]:
        return self._current_tokens
    
    def get_generator_history_embedding(self) -> List[float]:
        if self._generator_history_embedding == None:
            self._generator_history_embedding = get_history_embedding(
                self._sentences, self._generator_history_sentences
            )
        return self._generator_history_embedding             
    
    def get_narrative_memory_embedding(self) -> List[float]:
        if self._narrative_memory_embedding == None:
            self._narrative_memory_embedding = self._narrative_memory.get_state()
        return self._narrative_memory_embedding

# ============================================================
# History embedding
# ============================================================

def get_history_embedding(sentences: List[EncodedSentence], amounts: List[int]) -> List[float]:
    embedding = []
    pos: int = 0

    # large embeddings first
    for i in range(amounts[0]):
        embedding += sentences[pos].large_embedding
        pos += 1

    # medium embeddings next
    for i in range(amounts[1]):
        embedding += sentences[pos].medium_embedding
        pos += 1

    return embedding

# ============================================================
# ModelInput
# ============================================================

@dataclass(frozen=True)
class ModelInput:
    window: ContextWindow
    sequence_embedding: List[float]
    line_number: List[float]

# ============================================================
# InputEncoder
# ============================================================

@dataclass(frozen=True)
class InputEncoder:
    def encode(self, model_input: ModelInput) -> List[float]:
        history_embedding = model_input.window.get_generator_history_embedding()
        current_embedding = model_input.window.get_current_embedding()
        narrative_embedding = model_input.window.get_narrative_memory_embedding()
        sequence_embedding = model_input.sequence_embedding
        line_number = model_input.line_number
        
        return (
            history_embedding
            + current_embedding
            + narrative_embedding
            + sequence_embedding
            + line_number
        )
