# glp_network.py
from typing import List
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from uuid import uuid4

from .config import Configuration
from .tokens import Token, TokenPage, TokenLogit, TokenDictionary, LemmaGrammarDictionary, GRAMMAR_TOKENS
from .context import ModelInput, InputEncoder
from .neural import NeuralNetwork, ActivationMLP


# ============================================================
# TrainingSample
# ============================================================

@dataclass
class TrainingSample:
    input_vector: List[float]
    target: Token
    grammar: bool
    prev_lemma: str       


@dataclass
class TrainingBatch:
    samples: List[TrainingSample] = field(default_factory=list)


# ============================================================
# GLP Network (grammar + lemma)
# ============================================================

@dataclass
class GlpNetwork:
    configuration: Configuration
    token_dictionary: TokenDictionary
    encoder: InputEncoder = field(init=False)

    grammar_activation: ActivationMLP = field(init=False)
    grammar_network: NeuralNetwork = field(init=False)

    lemma_activation: ActivationMLP = field(init=False)
    lemma_network: NeuralNetwork = field(init=False)

    grammar_tokens: List[Token] = field(init=False)
    lemma_grammar_dict: LemmaGrammarDictionary = field(init=False)

    # paging structure remains
    page_list: List[TokenPage] = field(default_factory=list)
    pages: dict = field(default_factory=dict)

    def __post_init__(self):
        self.encoder = InputEncoder()
        self._create_grammar_network()
        self._create_lemma_network()

        self.grammar_tokens = [self.token_dictionary.map[text] for text in GRAMMAR_TOKENS]
        self.lemma_grammar_dict = LemmaGrammarDictionary()

    # --------------------------------------------------------
    # Network creation
    # --------------------------------------------------------

    def _create_grammar_network(self):
        self.grammar_activation = ActivationMLP(self.configuration.activation_hidden_size)
        self.grammar_network = NeuralNetwork(
            self.configuration.generator_input_size(),
            self.configuration.grammar_hidden_size,
            self.configuration.grammar_dimension,
            self.grammar_activation
        )

    def _create_lemma_network(self):
        self.lemma_activation = ActivationMLP(self.configuration.activation_hidden_size)
        self.lemma_network = NeuralNetwork(
            self.configuration.generator_input_size(),
            self.configuration.lemma_hidden_size,
            self.configuration.generator_output_size(),   # bits + tri + pages
            self.lemma_activation
        )

    # --------------------------------------------------------
    # Learning
    # --------------------------------------------------------

    def learn(self, model_input: ModelInput, target: Token, batch: TrainingBatch):
        """
        Add sample to batch. Paging still used to assign lemma tokens to pages,
        but no per-page networks anymore.
        """
        # Determine page for lemma tokens
        if not model_input.grammar:
            # update lemma grammar dictionary
            if target.is_lemma():
                idx = target.lemma_index
                g = model_input.window.last_token().grammar_index
                if idx is not None and g is not None:
                    mask = self.lemma_grammar_dict.mask.get(idx, 0)
                    mask |= (1 << g)
                    self.lemma_grammar_dict.set_mask(idx, mask)
            
            prev = model_input.window.last_lemma_token()   
            page = self.pages.get(prev)

            if page is None:
                # find existing page containing target
                page = next((p for p in self.page_list if p.has_output_token(target)), None)

                if page:
                    page.add_input_token(prev)
                elif len(self.page_list) < self.configuration.total_pages:
                    page = TokenPage(uuid4(), {prev}, [target])
                    self.page_list.append(page)
                else:
                    page = min(self.page_list, key=lambda p: p.input_size())

                self.pages[prev] = page

            if not page.has_output_token(target):
                page.add_output_token(target)

        # Add training sample
        batch.samples.append(
            TrainingSample(
                input_vector=self.encoder.encode(model_input),
                target=target,
                grammar=model_input.grammar,
                prev_lemma=model_input.window.last_lemma_token()   
            )
        )

    def learn_batch(self, batch: TrainingBatch):
        if not batch.samples:
            return

        # Split grammar vs lemma
        grammar_samples = [s for s in batch.samples if s.grammar]
        lemma_samples = [s for s in batch.samples if not s.grammar]

        if grammar_samples:
            self._learn_grammar(grammar_samples)

        if lemma_samples:
            self._learn_lemma(lemma_samples)

    # --------------------------------------------------------
    # Grammar learning
    # --------------------------------------------------------

    def _learn_grammar(self, samples: List[TrainingSample]):
        xs = torch.tensor([s.input_vector for s in samples], dtype=torch.float32)
        ys = torch.tensor([s.target.grammar_index for s in samples], dtype=torch.long)

        pred = self.grammar_network(xs)
        loss = F.cross_entropy(pred, ys)

        self.grammar_network.opt.zero_grad()
        loss.backward()
        self.grammar_network.opt.step()

    # --------------------------------------------------------
    # Lemma learning
    # --------------------------------------------------------

    def _learn_lemma(self, samples: List[TrainingSample]):
        xs = torch.tensor([s.input_vector for s in samples], dtype=torch.float32)
        pred = self.lemma_network(xs)

        # split output vector
        bits_dim = self.configuration.bits_dimension
        tri_dim = self.configuration.triple_dimension

        bits_pred = pred[:, :bits_dim]
        tri_pred = pred[:, bits_dim:bits_dim+tri_dim]
        page_pred = pred[:, bits_dim+tri_dim:]

        # targets
        bit_targets = []
        tri_targets = []
        page_targets = []

        for s in samples:
            idx = s.target.lemma_index

            # 1. bits
            bits = [(idx >> i) & 1 for i in range(bits_dim)]
            bit_targets.append(bits)

            # 2. tri-hot
            tri_bits = self.configuration.codebook.get_bits(idx)
            tri_vec = [0.0] * tri_dim
            for b in tri_bits:
                tri_vec[b] = 1.0
            tri_targets.append(tri_vec)

            # 3. page
            prev = s.prev_lemma
            page = self.pages.get(prev)

            if page is None:
                page_idx = 0   # fallback
            else:
                page_idx = self.page_list.index(page)

            page_targets.append(page_idx)

        bit_targets = torch.tensor(bit_targets, dtype=torch.float32)
        tri_targets = torch.tensor(tri_targets, dtype=torch.float32)
        page_targets = torch.tensor(page_targets, dtype=torch.long)

        # losses
        loss_bits = F.mse_loss(bits_pred, bit_targets)
        loss_tri = F.mse_loss(tri_pred, tri_targets)
        loss_page = F.cross_entropy(page_pred, page_targets)

        loss = loss_bits + loss_tri + loss_page

        self.lemma_network.opt.zero_grad()
        loss.backward()
        self.lemma_network.opt.step()

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    def select_grammar_token(self, token: Token, model_input: ModelInput) -> bool:
        last = model_input.window.last_token()
        if last.text == token.text:
            return False
        if token.is_eol():
            if not last.is_punctuation():
                return False
        elif last.is_punctuation():
              return False  
        if token.is_punctuation() and last.is_terminal():
            return False
        return True
    
    def propose(self, model_input: ModelInput):
        if model_input.grammar:
            return self._propose_grammar(model_input)
        return self._propose_lemma(model_input)

    def _propose_grammar(self, model_input: ModelInput):
        x = torch.tensor([self.encoder.encode(model_input)], dtype=torch.float32)
        with torch.no_grad():
            pred = self.grammar_network(x).squeeze(0)

        logits = []
        for tok in self.grammar_tokens:
            if self.select_grammar_token(tok, model_input):
                logit = float(pred[tok.grammar_index])
                logits.append(TokenLogit(tok, logit))

        logits.sort(key=lambda t: t.logit, reverse=True)
        return logits

    def _propose_lemma(self, model_input: ModelInput):
        x = torch.tensor([self.encoder.encode(model_input)], dtype=torch.float32)
        with torch.no_grad():
            pred = self.lemma_network(x).squeeze(0)

        bits_dim = self.configuration.bits_dimension
        tri_dim = self.configuration.triple_dimension

        bits_pred = pred[:bits_dim]
        tri_pred = pred[bits_dim:bits_dim+tri_dim]
        page_pred = pred[bits_dim+tri_dim:]

        # ------------------------------------------------------------
        # 1. Sort pages with logit
        # ------------------------------------------------------------
        page_scores = [(i, float(page_pred[i])) for i in range(len(self.page_list))]
        page_scores.sort(key=lambda t: t[1], reverse=True)

        grammar_index = model_input.window.last_token().grammar_index

        # ------------------------------------------------------------
        # 2. Iterate pages in order
        # ------------------------------------------------------------
        for page_idx, _ in page_scores:
            page = self.page_list[page_idx]

            logits = []

            # ------------------------------------------------------------
            # 3. Find valid lemmas
            # ------------------------------------------------------------
            for tok in page.output_tokens:
                idx = tok.lemma_index

                # check compatibility with grammar
                if grammar_index is not None:
                    if not self.lemma_grammar_dict.is_compatible(idx, grammar_index):
                        continue

                # check individual bits
                invalid_bits = 0
                for i in range(bits_dim):
                    expected = (idx >> i) & 1
                    predicted = bits_pred[i]
                    if expected == 0:
                        if predicted > 0.6:
                            invalid_bits += 1
                    elif predicted < 0.4:
                        invalid_bits += 1
                    
                # check if there are too many invalid bits
                if invalid_bits > 3:
                    continue

                # logit = average tri-hot activation
                tri_bits = self.configuration.codebook.get_bits(idx)
                invalid_tribits = 0
                for b in tri_bits:
                    if tri_pred[b] < 0.1:
                        invalid_tribits += 1
                if invalid_tribits >= 2:
                    continue 
                
                logit = float(sum(tri_pred[b] for b in tri_bits) / len(tri_bits))
                logits.append(TokenLogit(tok, logit))

            # ------------------------------------------------------------
            # 4. When page has overlap -> stop and return
            # ------------------------------------------------------------
            if logits:
                logits.sort(key=lambda t: t.logit, reverse=True)
                return logits

        # ------------------------------------------------------------
        # 5. There are no pages found with overlap
        # ------------------------------------------------------------
        return []
