# glp_network.py
from dataclasses import dataclass, field
from typing import List
from uuid import uuid4

import random
import torch
import torch.nn.functional as F

from .config import Configuration, ENABLE_PAGING, MAX_PAGELESS_VOCAB
from .tokens import (
    Token,
    TargetToken,
    TokenDictionary,
    TokenLogit,
    TokenPage,
    LemmaEmbeddingDictionary,
    GRAMMAR_TOKENS
)
from .sentence_encoder import SentenceEncoder
from .context import ModelInput, InputEncoder
from .neural import NeuralNetwork
from .training import TrainingSample, TrainingBatch
from .rulebased import RuleBasedFilter

RULE_BASED: RuleBasedFilter = RuleBasedFilter()

# ============================================================
# GlpNetwork
# ============================================================

@dataclass
class GlpNetwork:
    configuration: Configuration
    token_dictionary: TokenDictionary
    encoder: InputEncoder = field(init=False)

    glp_network: NeuralNetwork = field(init=False)

    grammar_tokens: List[Token] = field(init=False)
    lemma_embedding_dict: LemmaEmbeddingDictionary = field(init=False)
    sentence_encoder: SentenceEncoder = field(init=False)

    page_list: List[TokenPage] = field(default_factory=list)
    pages: dict = field(default_factory=dict)

    def __post_init__(self):
        self.encoder = InputEncoder()

        # monolithic network
        self.glp_network = NeuralNetwork(
            self.configuration.generator_input_size(),
            self.configuration.first_hidden_size,
            self.configuration.other_hidden_size,
            self.configuration.generator_output_size()
        )

        self.lemma_embedding_dict = self.create_lemma_embedding_dictionary()        
        self.grammar_tokens = [self.token_dictionary.map[text] for text in GRAMMAR_TOKENS]
        self.sentence_encoder = SentenceEncoder(self.lemma_embedding_dict, self.configuration)

    # ------------------------------------------------------------
    # Embedding Learning
    # ------------------------------------------------------------

    def create_lemma_embedding_dictionary(self) -> LemmaEmbeddingDictionary:
        return LemmaEmbeddingDictionary(
                self.configuration.lemma_input_dimension,
                self.configuration.lemma_output_dimension)

    def _update_lemma_embeddings_after_batch(self, samples: List[TrainingSample]):
        alpha: float = self.configuration.learn_alpha
        
        for s in samples:
            x = torch.tensor([s.input_vector], dtype=torch.float32)
            with torch.no_grad():
                pred = self.glp_network(x).squeeze(0)

            lemma_dim = self.configuration.lemma_output_dimension
            embedding_pred = pred[:lemma_dim]
            embedding_pred = F.normalize(embedding_pred, p=2, dim=0)

            emb_obj = self.lemma_embedding_dict.get_output_embedding(s.target)
            emb_obj.update(embedding_pred.tolist(), alpha)
            
    # ------------------------------------------------------------
    # Token Learning
    # ------------------------------------------------------------
    
    def learn(self, model_input: ModelInput, target: Token, batch: TrainingBatch):
        # TERMINAL GRAMMAR CASE
        if target.is_terminal():
            grammar_tok = target
            lemma_tok = target
        # NON-TERMINAL GRAMMAR CASE → ignore
        elif not target.is_lemma():
            return
        # LEMMA CASE
        else:           
            grammar_tok = model_input.window.last_token()
            lemma_tok = target
        tt = TargetToken(grammar=grammar_tok, lemma=lemma_tok)

        # find the page that has the input token
        prev_lemma = model_input.window.last_lemma_token()
        page = self.pages.get(prev_lemma)

        if page is None:
            # find a page that has the same target token
            max_input_size = self.configuration.max_page_input_size
            for p in self.page_list:
                if p.has_output_token(tt) and p.input_size() < max_input_size:
                    page = p
                    break

            if page:
                page.add_input_token(prev_lemma)

            # create a new page if possible
            elif len(self.page_list) < self.configuration.total_pages:
                page = TokenPage(uuid4(), {prev_lemma}, [tt])
                self.page_list.append(page)

            # use the page with the lowest amount of inputs
            else:
                page = min(self.page_list, key=lambda p: p.input_size())
                page.add_input_token(prev_lemma)

            self.pages[prev_lemma] = page

        if not page.has_output_token(tt):
            page.add_output_token(tt)

        page_index = self.page_list.index(page)

        sample = TrainingSample(
            input_vector=self.encoder.encode(model_input),
            target=tt,
            page_index=page_index
        )
        batch.samples.append(sample)

    def learn_batch(self, batch: TrainingBatch):
        if not batch.samples:
            return

        self._learn_glp(batch.samples)

        self._update_lemma_embeddings_after_batch(batch.samples)

    # ------------------------------------------------------------
    # Sample Learning
    # ------------------------------------------------------------
    
    def _learn_glp(self, samples: List[TrainingSample]):
        # Input batch
        xs = torch.tensor([s.input_vector for s in samples], dtype=torch.float32)

        # Predict only lemma embedding
        pred = self.glp_network(xs)

        # Normalize predicted embeddings
        pred = F.normalize(pred, p=2, dim=1)

        # Build target embeddings
        lemma_targets = []
        for s in samples:
            emb = self.lemma_embedding_dict.get_output_embedding(s.target).embedding
            lemma_targets.append(emb)

        lemma_targets = torch.tensor(lemma_targets, dtype=torch.float32)
        lemma_targets = F.normalize(lemma_targets, p=2, dim=1)

        # Cosine loss
        loss = 1.0 - F.cosine_similarity(pred, lemma_targets).mean()

        # Backprop
        self.glp_network.opt.zero_grad()
        loss.backward()
        self.glp_network.opt.step()

    # ------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------
    
    def propose(self, model_input: ModelInput):
        # Deterministic end of line after end punctuation
        if model_input.window.last_token().is_end_punctuation():
            return [TokenLogit(Token.EOL, Token.EOL, 1.0)]
        
        x = torch.tensor([self.encoder.encode(model_input)], dtype=torch.float32)
        with torch.no_grad():
            pred = self.glp_network(x).squeeze(0)
        lemma_pred = F.normalize(pred, p=2, dim=0)

        # Determine incompatible tokens
        min_tokens = self.configuration.min_words * 2
        incompatible_grammar = RULE_BASED.determine_incompatible_grammar(model_input, min_tokens)
        incompatible_lemma = RULE_BASED.determine_incompatible_lemma(model_input)

        # TARGET TOKENS: only from selected page
        page_pairs: List[TokenLogit] = []
        # This mode uses the graph for target routing like DMLG
        if ENABLE_PAGING:
            # PAGE ROUTING: use page graph transitions, default to first page with EOL
            last_lemma = model_input.window.last_lemma_token()
            page = self.pages.get(last_lemma, self.page_list[0])    
            output_tokens = page.output_tokens
        # This mode uses only the MLP over the full vocab
        else:
            if not hasattr(self, "output_tokens"):
                self.output_tokens = set()
                for p in self.page_list:
                    self.output_tokens.update(p.output_tokens)
                print(f"Built vocab of {len(self.output_tokens)} target tokens.")
            if MAX_PAGELESS_VOCAB < len(self.output_tokens):
                # Use random shuffle of vocab with period terminal token
                output_tokens = list(self.output_tokens)
                random.shuffle(output_tokens)
                output_tokens = output_tokens[:MAX_PAGELESS_VOCAB]
                output_tokens.append(TargetToken.PERIOD)
            else:
                # Use complete vocab in pageless mode
                output_tokens = self.output_tokens
                    
        # Create all eligible target token pairs
        for tok in output_tokens:
            if not tok.grammar.text in incompatible_grammar and \
               not tok.lemma.lower_text in incompatible_lemma:
                emb = torch.tensor(
                    self.lemma_embedding_dict.get_output_embedding(tok).embedding,
                    dtype=torch.float32,
                )                
                emb = F.normalize(emb, p=2, dim=0)
                cos = float(torch.dot(lemma_pred, emb))
                page_pairs.append(TokenLogit(grammar=tok.grammar, lemma=tok.lemma, logit=cos))

        # Sort pairs and return
        page_pairs.sort(key=lambda t: t.logit, reverse=True)
        return page_pairs
