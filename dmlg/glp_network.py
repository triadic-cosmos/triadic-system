# glp_network.py
from typing import List
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from uuid import uuid4

from .config import Configuration
from .tokens import Token, TokenPage, TokenLogit, TokenDictionary, LemmaGrammarDictionary, LemmaEmbeddingDictionary, GRAMMAR_TOKENS
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

    lemma_embedding_dict: LemmaEmbeddingDictionary = field(init=False)

    # paging structure remains
    page_list: List[TokenPage] = field(default_factory=list)
    pages: dict = field(default_factory=dict)

    def __post_init__(self):
        self.encoder = InputEncoder()
        self._create_grammar_network()
        self._create_lemma_network()
        
        self.lemma_embedding_dict = LemmaEmbeddingDictionary(self.configuration.lemma_dimension)

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
            alpha = self._current_alpha(batch.training_count)
            self._update_lemma_embeddings_after_batch(lemma_samples, alpha)

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
        lemma_dim = self.configuration.lemma_dimension
        page_dim = self.configuration.total_pages

        embedding_pred = pred[:, :lemma_dim] # learnable embedding prediction
        page_pred = pred[:, lemma_dim:lemma_dim+page_dim]  # page one-hot logits

        # targets
        embedding_targets = []
        page_targets = []

        for s in samples:
            # 1. lemma embedding target
            lemma_emb = self.lemma_embedding_dict.get_embedding(s.target)
            embedding_targets.append(lemma_emb.embedding)

            # 2. page one-hot target
            page = self.pages.get(s.prev_lemma)
            if page is None:
                page_idx = 0
            else:
                page_idx = self.page_list.index(page)

            page_targets.append(page_idx)

        embedding_targets = torch.tensor(embedding_targets, dtype=torch.float32)
        page_targets = torch.tensor(page_targets, dtype=torch.long)

        # normalize predicted embeddings
        embedding_pred = F.normalize(embedding_pred, p=2, dim=1)
        embedding_targets = F.normalize(embedding_targets, p=2, dim=1)

        # losses
        loss_embedding = 1.0 - F.cosine_similarity(embedding_pred, embedding_targets).mean()
        loss_page = F.cross_entropy(page_pred, page_targets)

        loss = loss_embedding + loss_page

        self.lemma_network.opt.zero_grad()
        loss.backward()
        self.lemma_network.opt.step()

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
            # 1. Run inference to get model embedding prediction
            x = torch.tensor([s.input_vector], dtype=torch.float32)
            with torch.no_grad():
                pred = self.lemma_network(x).squeeze(0)

            lemma_dim = self.configuration.lemma_dimension
            embedding_pred = pred[:lemma_dim]

            # normalize predicted embedding
            embedding_pred = F.normalize(embedding_pred, p=2, dim=0)

            # 2. Get the learnable embedding object
            lemma_emb = self.lemma_embedding_dict.get_embedding(s.target)

            # 3. Apply alpha‑mix update
            lemma_emb.update(embedding_pred.tolist(), alpha)

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

        lemma_dim = self.configuration.lemma_dimension
        page_dim = self.configuration.total_pages

        # ------------------------------------------------------------
        # 1. Split output: embedding + page logits
        # ------------------------------------------------------------
        embedding_pred = pred[:lemma_dim]
        page_pred = pred[lemma_dim:lemma_dim+page_dim]

        # normalize model embedding
        embedding_pred = F.normalize(embedding_pred, p=2, dim=0)

        # ------------------------------------------------------------
        # 2. Sort pages by logit
        # ------------------------------------------------------------
        page_scores = [(i, float(page_pred[i])) for i in range(len(self.page_list))]
        page_scores.sort(key=lambda t: t[1], reverse=True)

        grammar_index = model_input.window.last_token().grammar_index

        # ------------------------------------------------------------
        # 3. Iterate pages in descending score order
        # ------------------------------------------------------------
        for page_idx, page_logit in page_scores:
            page = self.page_list[page_idx]
            logits = []

            # ------------------------------------------------------------
            # 4. Evaluate tokens in this page
            # ------------------------------------------------------------
            for tok in page.output_tokens:
                idx = tok.lemma_index

                # grammar compatibility filter
                if grammar_index is not None:
                    if not self.lemma_grammar_dict.is_compatible(idx, grammar_index):
                        continue

                # get lemma embedding
                lemma_emb = self.lemma_embedding_dict.get_embedding(tok)
                emb = torch.tensor(lemma_emb.embedding, dtype=torch.float32)
                emb = F.normalize(emb, p=2, dim=0)

                # cosine similarity
                cos = float(torch.dot(embedding_pred, emb))

                # final score = cosine similarity (page already sorted)
                logits.append(TokenLogit(tok, cos))

            # ------------------------------------------------------------
            # 5. If this page yields valid tokens → return them
            # ------------------------------------------------------------
            if logits:
                logits.sort(key=lambda t: t.logit, reverse=True)
                return logits

        # ------------------------------------------------------------
        # 6. No page produced valid tokens
        # ------------------------------------------------------------
        return []
