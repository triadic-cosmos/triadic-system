# context.py
from dataclasses import dataclass, field
from typing import List

from .tokens import Token
from .sentence_encoder import EncodedSentence
from .narrative_memory import NarrativeMemory
from .config import Configuration

# ============================================================
# ContextWindow
# ============================================================

@dataclass
class ContextWindow:
    configuration: Configuration
    lemma_embedding_dict: any 

    def __post_init__(self):
        empty_sentence = EncodedSentence.make_empty_sentence(self.configuration)
        self.clear_current_sentence()
        self._last_position = self.configuration.content_max_lemmas - 1        
        self._sentences = [empty_sentence] * self.configuration.context_max_sentences
        self._generator_history_embedding = None
        self._generator_history_sentences = self.configuration.generator_history_sentences
        self._narrative_memory = NarrativeMemory(self.lemma_embedding_dict, self.configuration)
        self._narrative_memory_embedding = None
        
    def clear_current_sentence(self):
        self._position = 0

        # USE MEDIUM EMBEDDING SIZE (2 floats)
        cut = self.configuration.sentence_medium_embedding_size

        self._current_embedding = [
            [0.0] * cut for _ in range(self.configuration.content_max_lemmas)
        ]

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
        lemma_text = self._last_lemma.text
        last_text = self._last_token.text
        if lemma_text == last_text:
            return lemma_text
        return lemma_text + last_text

    def add_token(self, token: Token):
        self._forelast_token = self._last_token
        self._last_token = token

        if token.is_eol():
            self._last_lemma = token

        elif token.is_lemma():
            self._last_lemma = token

            if self._position <= self._last_position:
                emb = self.lemma_embedding_dict.get_input_embedding(token).embedding

                # CLIP TO MEDIUM SIZE (2 floats)
                cut = self.configuration.sentence_medium_embedding_size
                vec = emb[:cut]

                self._current_embedding[self._position] = vec
                self._position += 1

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
            
    def is_filled(self):
        return self._sentences[len(self._sentences) - 1] != EMPTY_SENTENCE

    def get_current_embedding(self) -> List[float]:
        cur_pos = min(1.0, self._position / self.configuration.content_max_lemmas)
        flat = []
        for vec in self._current_embedding:
            flat.extend(vec)
        return [cur_pos] + flat

    def copy_current(self) -> "ContextWindow":
        ctx: ContextWindow = ContextWindow(self.configuration, self.lemma_embedding_dict)
        ctx._position = self._position
        ctx._current_embedding = [vec.copy() for vec in self._current_embedding]
        ctx._last_token = self._last_token
        ctx._forelast_token = self._forelast_token
        ctx._last_lemma = self._last_lemma
        ctx._sentences = self._sentences
        ctx._generator_history_embedding = self._generator_history_embedding
        return ctx
        
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
        
        return (
            history_embedding
            + current_embedding
            + narrative_embedding
            + sequence_embedding
        )
