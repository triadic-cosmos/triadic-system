# glp_network.py
from dataclasses import dataclass, field
from typing import List
from uuid import uuid4

import torch
import torch.nn.functional as F

from .config import Configuration
from .tokens import (
    Token,
    TargetToken,
    TokenDictionary,
    TokenLogit,
    TokenPage,
    LemmaEmbeddingDictionary,
    GRAMMAR_TOKENS,
    NO_PUNCTUATION_TOKENS,
    CONJUGATION_TOKENS,
    NOUN_TOKENS
)
from .sentence_encoder import SentenceEncoder
from .context import ModelInput, InputEncoder
from .neural import NeuralNetwork, ActivationMLP

# ============================================================
# TrainingSample
# ============================================================

@dataclass
class TrainingSample:
    input_vector: List[float]
    target: TargetToken      # grammar + lemma
    page_index: int          # -1 for terminals

# ============================================================
# TrainingBatch
# ============================================================

@dataclass
class TrainingBatch:
    samples: List[TrainingSample] = field(default_factory=list)
    training_count: int = 0

# ============================================================
# GlpNetwork
# ============================================================

@dataclass
class GlpNetwork:
    configuration: Configuration
    token_dictionary: TokenDictionary
    encoder: InputEncoder = field(init=False)

    glp_activation: ActivationMLP = field(init=False)
    glp_network: NeuralNetwork = field(init=False)

    grammar_tokens: List[Token] = field(init=False)
    lemma_embedding_dict: LemmaEmbeddingDictionary = field(init=False)
    sentence_encoder: SentenceEncoder = field(init=False)

    page_list: List[TokenPage] = field(default_factory=list)
    pages: dict = field(default_factory=dict)

    def __post_init__(self):
        self.encoder = InputEncoder()

        # monolithic network
        self.glp_activation = ActivationMLP(self.configuration.activation_hidden_size)
        self.glp_network = NeuralNetwork(
            self.configuration.generator_input_size(),
            self.configuration.glp_hidden_size,
            self.configuration.generator_output_size(),
            self.glp_activation,
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

    # reduce alpha to train embeddings in function of number of epochs
    def _current_alpha(self, training_count: int) -> float:
        alpha_max = self.configuration.learn_alpha
        alpha_min = alpha_max / self.configuration.alpha_damping
        T = self.configuration.max_alpha_transitions

        if training_count >= T:
            return alpha_min

        factor = 1.0 - (training_count / T)  # from 1 to 0
        return alpha_min + (alpha_max - alpha_min) * factor

    def _update_lemma_embeddings_after_batch(self, samples: List[TrainingSample], alpha: float):
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
    # Learning
    # ------------------------------------------------------------
    def learn(self, model_input: ModelInput, target: Token, batch: TrainingBatch):
        # TERMINAL CASE
        if target.is_terminal():
            tt = TargetToken(grammar=target, lemma=target)
            sample = TrainingSample(
                input_vector=self.encoder.encode(model_input),
                target=tt,
                page_index=-1
            )
            batch.samples.append(sample)
            return

        # NON‑TERMINAL GRAMMAR → negeren
        if not target.is_lemma():
            return

        # LEMMA CASE
        grammar_tok = model_input.window.last_token()
        lemma_tok = target
        tt = TargetToken(grammar=grammar_tok, lemma=lemma_tok)

        prev_lemma = model_input.window.last_lemma_token()
        page = self.pages.get(prev_lemma)

        if page is None:
            page = next((p for p in self.page_list if p.has_output_token(tt)), None)

            if page:
                page.add_input_token(prev_lemma)

            elif len(self.page_list) < self.configuration.total_pages:
                page = TokenPage(uuid4(), {prev_lemma}, [tt])
                self.page_list.append(page)

            else:
                page = min(self.page_list, key=lambda p: p.input_size())

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

        alpha = self._current_alpha(batch.training_count)
        self._update_lemma_embeddings_after_batch(batch.samples, alpha)

    # ------------------------------------------------------------
    # Monolithic learning
    # ------------------------------------------------------------
    def _learn_glp(self, samples: List[TrainingSample]):
        xs = torch.tensor([s.input_vector for s in samples], dtype=torch.float32)
        pred = self.glp_network(xs)

        lemma_dim = self.configuration.lemma_output_dimension
        page_dim = self.configuration.total_pages

        lemma_pred = pred[:, :lemma_dim]
        page_pred = pred[:, lemma_dim:]

        # Targets
        lemma_targets_list = []
        page_targets_list = []
        page_mask_list = []  # True = CE (lemma tokens), False = MSE (terminals)

        for s in samples:
            emb = self.lemma_embedding_dict.get_output_embedding(s.target).embedding
            lemma_targets_list.append(emb)

            if s.page_index == -1:
                zero = [0.0] * page_dim
                page_targets_list.append(zero)
                page_mask_list.append(False)
            else:
                onehot = [0.0] * page_dim
                onehot[s.page_index] = 1.0
                page_targets_list.append(onehot)
                page_mask_list.append(True)

        lemma_targets = torch.tensor(lemma_targets_list, dtype=torch.float32)
        page_targets = torch.tensor(page_targets_list, dtype=torch.float32)
        page_mask = torch.tensor(page_mask_list, dtype=torch.bool)

        # Normalize lemma embeddings
        lemma_pred = F.normalize(lemma_pred, p=2, dim=1)
        lemma_targets = F.normalize(lemma_targets, p=2, dim=1)

        # Lemma loss (cosine)
        loss_lemma = 1.0 - F.cosine_similarity(lemma_pred, lemma_targets).mean()

        # Page loss
        if page_mask.any():
            ce_pred = page_pred[page_mask]
            ce_target = page_targets[page_mask].argmax(dim=1)
            loss_page_ce = F.cross_entropy(ce_pred, ce_target)
        else:
            loss_page_ce = torch.tensor(0.0, dtype=torch.float32)

        if (~page_mask).any():
            mse_pred = page_pred[~page_mask]
            mse_target = page_targets[~page_mask]
            loss_page_mse = F.mse_loss(mse_pred, mse_target)
        else:
            loss_page_mse = torch.tensor(0.0, dtype=torch.float32)

        loss_page = loss_page_ce + loss_page_mse

        loss = loss_lemma + loss_page

        self.glp_network.opt.zero_grad()
        loss.backward()
        self.glp_network.opt.step()

    # ------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------
    def determine_incompatible_grammar(self, model_input: ModelInput) -> set:
        last_grammar = model_input.window.last_grammar_token()
        if last_grammar.text in CONJUGATION_TOKENS:
            return CONJUGATION_TOKENS
        if last_grammar.text in NOUN_TOKENS:
            return NOUN_TOKENS
        return set()
            
    def select_terminal_token(self, target: TargetToken, model_input: ModelInput) -> bool:
        grammar = target.grammar
        last = model_input.window.last_token()

        # 1. Prevent repeating the same terminal
        if last.text == grammar.text:
            return False

        # 2. EOL rules
        if grammar.is_eol():
            if not last.is_end_punctuation():
                return False
            return True

        # 3. Punctuation rules
        if grammar.is_all_punctuation():
            if last.is_all_punctuation():
                return False
            if last.is_terminal():
                return False
            forelast = model_input.window.forelast_token()
            if forelast.text in NO_PUNCTUATION_TOKENS:
                return False
            return True

        # 4. Default: allowed
        return True

    def propose(self, model_input: ModelInput):
        # deterministic end of line after end punctuation
        if model_input.window.last_token().is_end_punctuation():
            return [TokenLogit(Token.EOL, Token.EOL, 1.0)]
        
        x = torch.tensor([self.encoder.encode(model_input)], dtype=torch.float32)
        with torch.no_grad():
            pred = self.glp_network(x).squeeze(0)

        lemma_dim = self.configuration.lemma_output_dimension
        page_dim = self.configuration.total_pages

        # split output
        lemma_pred = F.normalize(pred[:lemma_dim], p=2, dim=0)
        page_pred = pred[lemma_dim : ]

        # --- PAGE ROUTING ---
        page_scores = [(i, float(page_pred[i])) for i in range(len(self.page_list))]
        page_scores.sort(key=lambda t: t[1], reverse=True)

        # TERMINAL TOKENS: implicitly present in each page
        terminal_pairs: List[TokenLogit] = []
        for terminal in TargetToken.TERMINALS:
            if self.select_terminal_token(terminal, model_input):
                emb = torch.tensor(
                    self.lemma_embedding_dict.get_output_embedding(terminal).embedding,
                    dtype=torch.float32,
                )
                emb = F.normalize(emb, p=2, dim=0)
                cos = float(torch.dot(lemma_pred, emb))
                terminal_pairs.append(
                    TokenLogit(grammar=terminal.grammar, lemma=terminal.lemma, logit=cos)
                )

        incompatible_grammar = self.determine_incompatible_grammar(model_input)

        # --- PAGE LOOP ---
        for page_idx, _ in page_scores:
            page = self.page_list[page_idx]
            page_pairs: List[TokenLogit] = []
                    
            # LEMMA TOKENS: from page, paired with grammar_top
            for tok in page.output_tokens:
                if not tok.grammar.text in incompatible_grammar:
                    emb = torch.tensor(
                        self.lemma_embedding_dict.get_output_embedding(tok).embedding,
                        dtype=torch.float32,
                    )                
                    emb = F.normalize(emb, p=2, dim=0)
                    cos = float(torch.dot(lemma_pred, emb))
                    page_pairs.append(TokenLogit(grammar=tok.grammar, lemma=tok.lemma, logit=cos))

            # when page has valid pairs → sort and STOP
            if page_pairs:
                page_pairs.sort(key=lambda t: t.logit, reverse=True)
                return page_pairs + terminal_pairs # only add terminal pairs here

        # --- FALLBACK: no valid pairs found in any of the pages ---
        return None
