# glp_network.py
from dataclasses import dataclass, field
from typing import List
from uuid import uuid4

import random
import torch
import torch.nn.functional as F

from .config import Configuration
from .writer_story import WriterParams
from .context import ModelInput, InputEncoder
from .neural import NeuralNetwork
from .training import TrainingSampleSentence
from .rulebased import RuleBasedFilter
from .tokens import (
    Token,
    TargetToken,
    TokenLogit,
    TokenPage,
    LemmaEmbeddingDictionary
)

RULE_BASED: RuleBasedFilter = RuleBasedFilter()

# ============================================================
# Evaluation
# ============================================================

@dataclass
class Evaluation:
    target: TargetToken
    candidates: int
    matrix: float
    mlp: float
    rank_matrix: int
    rank_mlp: int
    agreement: float
    allowed: bool
    grammar_blocked: bool    
    lemma_blocked: bool

# ============================================================
# GlpNetwork
# ============================================================

@dataclass
class GlpNetwork:
    configuration: Configuration
    encoder: InputEncoder = field(init=False)

    glp_network: NeuralNetwork = field(init=False)

    lemma_embedding_dict: LemmaEmbeddingDictionary = field(init=False)

    page_list: List[TokenPage] = field(default_factory=list)
    pages: dict = field(default_factory=dict)

    def __post_init__(self):
        self.encoder = InputEncoder()
        self.lemma_embedding_dict = self.create_lemma_embedding_dictionary()        

        # Input vector dimensions
        token_input_size = self.configuration.generator_token_input_size()
        token_pred_input_size = token_input_size + self.configuration.token_embedding_dimension
        
        # MLP network
        self.glp_network = NeuralNetwork(
            token_pred_input_size,
            self.configuration.first_hidden_size,
            self.configuration.other_hidden_size,
            self.configuration.generator_output_size()
        )

        # Predictor matrix
        self.predictor_matrix = torch.empty(
            self.configuration.generator_output_size(),
            token_input_size
        )
        torch.nn.init.normal_(self.predictor_matrix, mean=0.0, std=0.01)
                
    # ------------------------------------------------------------
    # Token conversion
    # ------------------------------------------------------------

    def get_target_tokens(self) -> set:
        output_tokens = set()
        for p in self.page_list:
            output_tokens.update(p.output_tokens)
        return output_tokens

    def add_to_page(self, tt: TargetToken, prev_lemma: str):
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

    def convert_to_target_tokens(self, tokens: List[Token]) -> List[TargetToken]:
        previous_token = Token.EOL
        grammar_token = None
        target_tokens = []
        
        for token in tokens:
            if token.is_terminal():
                if grammar_token:
                    print(f"Grammar token without lemma {grammar_token}!")
                    grammar_token = None
                # Do not convert end of line token
                if token.is_eol():
                    continue
                target_token = TargetToken(token, token)                 
                target_tokens.append(target_token)
                self.add_to_page(target_token, previous_token.text)
                previous_token = token
            elif token.is_grammar():
                if grammar_token:
                    print(f"Grammar token without lemma {grammar_token}!")
                grammar_token = token
            elif not grammar_token:
                print(f"Lemma token without grammar {token}!")
            else:
                target_token: TargetToken = TargetToken(grammar_token, token)
                target_tokens.append(target_token)
                self.add_to_page(target_token, previous_token.text)
                previous_token = token
                grammar_token = None

        return target_tokens

    def create_lemma_embedding_dictionary(self) -> LemmaEmbeddingDictionary:
        return LemmaEmbeddingDictionary(self.configuration.token_embedding_dimension)

    # ------------------------------------------------------------
    # Model merging
    # ------------------------------------------------------------

    @staticmethod
    def merge_predictor_matrices(
        matrices: list[torch.Tensor],
        weights: list[float]
    ) -> torch.Tensor:

        if len(matrices) < 2:
            raise ValueError("At least two matrices are needed")

        if len(matrices) != len(weights):
            raise ValueError(
                "Number of matrices and weights must match"
            )

        total = sum(weights)

        if total == 0:
            raise ValueError("Weights sum to zero")

        # normalize
        weights = [w / total for w in weights]

        shape = matrices[0].shape

        for m in matrices[1:]:
            if m.shape != shape:
                raise ValueError(
                    "All predictor matrices must have equal shape"
                )

        merged = torch.zeros_like(matrices[0])

        for w, m in zip(weights, matrices):
            merged += w * m

        return merged

    @staticmethod
    def merge_networks(
        configuration: Configuration,
        networks: list["GlpNetwork"],
        weights: list[float]
    ) -> "GlpNetwork":
        merged: GlpNetwork = GlpNetwork(configuration)

        # Merge predictor matrices
        matrix = [network.predictor_matrix for network in networks]        
        merged.predictor_matrix = GlpNetwork.merge_predictor_matrices(matrix, weights)

        # Merge neural networks
        neural = [network.glp_network for network in networks]
        merged.glp_network = NeuralNetwork.merge(neural, weights)

        # Merge pages
        for network in networks:
            for page in network.page_list:
                for input_token in page.input_tokens:
                    for output_token in page.output_tokens:
                        merged.add_to_page(output_token, input_token)
                        
        return merged

    # ------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------
    
    def learn_sentence_matrix(self, sample: TrainingSampleSentence):
        # Configuration
        learning_rate = self.configuration.matrix_learning_rate
        decay = self.configuration.matrix_decay
        
        # Initialize model input
        model_input = sample.model_input.copy()
        model_input.window.start_sentence()
        
        # Update prediction matrix
        with torch.no_grad():
            for target in sample.target_tokens:
                # Input vector
                encoded_input = self.encoder.encode_token(model_input)
                model_input.window.add_target_token(target)                
                input_vec = torch.tensor([encoded_input], dtype=torch.float32)
                input_vec = F.normalize(input_vec, p=2, dim=1)

                # Output vector
                embedding = self.lemma_embedding_dict.get_embedding(target).embedding
                output_vec = torch.tensor([embedding], dtype=torch.float32)
                output_vec = F.normalize(output_vec, p=2, dim=1)
                                
                # Calculate delta
                delta_matrix = (
                    learning_rate *
                    torch.outer(output_vec.squeeze(), input_vec.squeeze())
                )

                # Update matrix
                self.predictor_matrix = delta_matrix + self.predictor_matrix * decay

            self.predictor_matrix = self.predictor_matrix.detach()

    def learn_sentence_mlp(self, sample: TrainingSampleSentence):
        # Initialize model input
        model_input = sample.model_input.copy()
        model_input.window.start_sentence()

        # Build inputs and outputs from target tokens
        encoded_inputs = []
        encoded_outputs = []

        for target in sample.target_tokens:
            encoded_input = self.encoder.encode_token(model_input)
            model_input.window.add_target_token(target)
            
            # Add prediction of matrix to input vector for MLP
            matrix_input = torch.tensor([encoded_input], dtype=torch.float32).squeeze()
            matrix_pred = self.predictor_matrix @ matrix_input
            matrix_pred = F.normalize(matrix_pred, p=2, dim=0)      
            encoded_input += matrix_pred.tolist()
            encoded_inputs.append(encoded_input)

            embedding = self.lemma_embedding_dict.get_embedding(target).embedding
            encoded_outputs.append(embedding)
        
        # Train MLP network
        mlp_input = torch.tensor([encoded_inputs], dtype=torch.float32)
        mlp_input = F.normalize(mlp_input, p=2, dim=1)
        mlp_output = torch.tensor([encoded_outputs], dtype=torch.float32)
        mlp_output = F.normalize(mlp_output, p=2, dim=1)
        mlp_pred = self.glp_network(mlp_input)
        mlp_pred = F.normalize(mlp_pred, p=2, dim=1)
        cos_loss = 1.0 - F.cosine_similarity(mlp_pred, mlp_output).mean()
        l1_loss = torch.mean(torch.abs(mlp_pred - mlp_output))
        loss = cos_loss + l1_loss
        self.glp_network.opt.zero_grad()
        loss.backward()
        self.glp_network.opt.step()

    # ------------------------------------------------------------
    # Evaluate model with a curriculum sentence
    # ------------------------------------------------------------

    def evaluate(self, sample: TrainingSampleSentence) -> List[Evaluation]:
        evaluations = []
        model_input = sample.model_input.copy()
        model_input.window.start_sentence()

        for target in sample.target_tokens:
            # Encode input and advance context
            page: TokenPage = self.pages.get(model_input.window.last_token().text, self.page_list[0])            
            encoded_input = self.encoder.encode_token(model_input)
            incompatible_grammar = RULE_BASED.determine_incompatible_grammar(model_input, 1)
            incompatible_lemma = RULE_BASED.determine_incompatible_lemma(model_input)                        
            model_input.window.add_target_token(target)

            # Skip a fully deterministic choice
            if len(page.output_tokens) <= 1:
                continue            
 
            # Do rule-based filtering
            filtered_candidates = []
            # Prevent filtering of target
            for tok in page.output_tokens:
                if tok == target or \
                   (tok.grammar.text not in incompatible_grammar and \
                    tok.lemma.lower_text not in incompatible_lemma):
                    filtered_candidates.append(tok)
                    
            # Skip very small and very large candidates sets        
            num_candidates = len(filtered_candidates)
            if num_candidates <= 2 or num_candidates > self.configuration.evaluation_max_candidates:
                continue
            
            # Infer proposal
            with torch.no_grad():
                matrix_input = torch.tensor([encoded_input], dtype=torch.float32).squeeze()
                matrix_pred = self.predictor_matrix @ matrix_input
                matrix_pred = F.normalize(matrix_pred, p=2, dim=0)                
                mlp_input = torch.tensor([encoded_input + matrix_pred.tolist()], dtype=torch.float32)
                mlp_pred = self.glp_network(mlp_input).squeeze(0)
                mlp_pred = F.normalize(mlp_pred, p=2, dim=0)
                agreement = torch.dot(matrix_pred, mlp_pred)

            # Score all output tokens
            matrix_logits: List[float] = []
            mlp_logits: List[float] = []
            target_logits: List[float] = None
            
            # Check all possible output tokens for page
            for tok in filtered_candidates:
                emb = torch.tensor(
                    self.lemma_embedding_dict.get_embedding(tok).embedding,
                    dtype=torch.float32,
                )                
                emb = F.normalize(emb, p=2, dim=0)
                
                matrix_logit = score(emb, matrix_pred)                    
                mlp_logit = score(emb, mlp_pred)
                
                if tok == target:
                    # target candidate logits
                    target_logits = [matrix_logit, mlp_logit]
                else:
                    # other candidate logits
                    matrix_logits.append(matrix_logit)
                    mlp_logits.append(mlp_logit)

            # Create evaluation
            grammar_blocked = target.grammar.text in incompatible_grammar
            lemma_blocked = target.lemma.lower_text in incompatible_lemma
            allowed = not grammar_blocked and not lemma_blocked
            if matrix_logit and mlp_logits:
                matrix_logits.sort(reverse=True)
                mlp_logits.sort(reverse=True)
                evaluations.append(Evaluation(
                    target, len(filtered_candidates),
                    target_logits[0] - matrix_logits[0],
                    target_logits[1] - mlp_logits[0],
                    rank(matrix_logits, target_logits[0]),
                    rank(mlp_logits, target_logits[1]),
                    agreement, allowed, grammar_blocked, lemma_blocked))
            else:
                evaluations.append(Evaluation(
                    target, len(filtered_candidates), 0, 0, 1, 1,
                    agreement, allowed, grammar_blocked, lemma_blocked))
                 
        return evaluations
        
    # ------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------

    def propose(self, params: WriterParams, model_input: ModelInput):
        # Deterministic end of line after end punctuation
        if model_input.window.last_token().is_end_punctuation():
            return [TokenLogit(TargetToken.EOL, 1.0)]
        
        # Determine incompatible tokens
        min_tokens = params.min_words
        incompatible_grammar = RULE_BASED.determine_incompatible_grammar(model_input, min_tokens)
        incompatible_lemma = RULE_BASED.determine_incompatible_lemma(model_input)

        # TARGET TOKENS: only from selected page
        page_pairs: List[TokenLogit] = []
        # This mode uses the graph for target routing like DMLG
        if params.enable_paging:
            # PAGE ROUTING: use page graph transitions, default to first page with EOL
            page = self.pages.get(model_input.window.last_token().text, self.page_list[0])
            output_tokens = page.output_tokens
        # This mode uses only the MLP over the full vocab
        else:
            if not hasattr(self, "output_tokens"):
                self.output_tokens = self.get_target_tokens()                
                print(f"Built vocab of {len(self.output_tokens)} target tokens.")
            if params.max_pageless_vocab < len(self.output_tokens):
                # Use random shuffle of vocab with period terminal token
                output_tokens = list(self.output_tokens)
                random.shuffle(output_tokens)
                output_tokens = output_tokens[:params.max_pageless_vocab]
                output_tokens.append(TargetToken.PERIOD)
            else:
                # Use complete vocab in pageless mode
                output_tokens = self.output_tokens
        
        # Infer proposal 
        with torch.no_grad():
            matrix_input = torch.tensor([self.encoder.encode_token(model_input)], dtype=torch.float32)
            matrix_pred = self.predictor_matrix @ matrix_input.squeeze()
            matrix_pred = F.normalize(matrix_pred, p=2, dim=0)
            mlp_input = torch.tensor([self.encoder.encode_token(model_input) + matrix_pred.tolist()], dtype=torch.float32)
            mlp_pred = self.glp_network(mlp_input).squeeze(0)
            mlp_pred = F.normalize(mlp_pred, p=2, dim=0)
        
        # Create all eligible target token pairs
        for tok in output_tokens:
            if not tok.grammar.text in incompatible_grammar and \
               not tok.lemma.lower_text in incompatible_lemma:
                emb = torch.tensor(
                    self.lemma_embedding_dict.get_embedding(tok).embedding,
                    dtype=torch.float32,
                )                
                emb = F.normalize(emb, p=2, dim=0)
                logit = score(emb, mlp_pred)
                page_pairs.append(TokenLogit(token=tok, logit=logit))

        # Sort pairs and return
        page_pairs.sort(key=lambda t: t.logit, reverse=True)
        return page_pairs

# Scoring function
def score(embedding, prediction) -> float:
    cos_score = torch.dot(prediction, embedding)
    l1_score = torch.mean(torch.abs(prediction - embedding))
    return cos_score - l1_score

# Determine rank
def rank(logits, logit) -> int:
    return sum(l > logit for l in logits) + 1
