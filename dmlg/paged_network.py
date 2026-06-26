# paged_network.py
from typing import List
from dataclasses import dataclass, field
from uuid import uuid4, UUID
import torch.nn.functional as F
import torch
import math

from .config import Configuration
from .tokens import Token, TokenPage, TokenLogit
from .context import ModelInput, InputEncoder
from .neural import NeuralNetwork, ActivationMLP

@dataclass
class TrainingSample:
    page: TokenPage
    input_vector: List[float]
    target: Token

    target_index: int = field(init=False)

    def __post_init__(self):
        self.target_index = self.page.get_output_index(self.target)

@dataclass
class TrainingBatch:
    samples: List[TrainingSample] = field(default_factory=list)

@dataclass
class PredictionNetwork:
    page: TokenPage
    network: NeuralNetwork
    encoder: InputEncoder
    configuration: Configuration

    def __post_init__(self):
        self.opt = torch.optim.Adam(self.network.parameters(), lr=1e-3)

    def learn_prebatch(self, model_input: ModelInput, target: Token, batch: TrainingBatch):
        batch.samples.append(TrainingSample(self.page, self.encoder.encode(model_input), target))

    def learn_batch(self, samples: List[TrainingSample]):
        # 1. Batch input
        xs = torch.tensor([s.input_vector for s in samples], dtype=torch.float32)

        # 2. Forward pass
        self.opt.zero_grad()
        pred = self.network(xs) # shape: (B, DIMENSION)

        losses = []
        output_dimension = self.configuration.output_dimension

        for i, sample in enumerate(samples):
            L = pred[i]  # (DIMENSION,)
            page = sample.page

            active = list(range(len(page.output_tokens)))  # N tokens

            # 3. Bit-matrix for all tokens in page
            #    bv. [[3], [7], [12, 25, 33], ...]
            bits_list = [self.configuration.codebook.get_bits(idx) for idx in active]

            # 4. Vectorized decode:
            #    - mask of shape (N, DIMENSION)
            #    - mask[j, b] = 1 if bit b is in token j
            N = len(active)
            mask = torch.zeros((N, output_dimension), dtype=torch.float32)

            for j, bits in enumerate(bits_list):
                mask[j, bits] = 1.0

            # 5. Token-logits = average of base logits
            #    shape: (N,)
            token_logits = (mask @ L) / mask.sum(dim=1)

            # 6. CE-loss
            target_pos = sample.target_index
            loss_i = torch.nn.functional.cross_entropy(
                token_logits.unsqueeze(0),
                torch.tensor([target_pos], dtype=torch.long)
            )

            losses.append(loss_i)

        # 7. Backprop
        loss = torch.stack(losses).mean()
        loss.backward()
        self.opt.step()

        return loss.item()
        
    def propose(self, model_input: ModelInput):
        if self.page.output_size() == 1:
            return [TokenLogit(self.page.output_tokens[0], 1.0)]

        encoded_input = self.encoder.encode(model_input)
        x = torch.tensor([encoded_input], dtype=torch.float32)

        with torch.no_grad():
            L = self.network(x).squeeze(0) 

        logits: List[TokenLogit] = []

        for tok in self.page.output_tokens:
            if not self.select_token(tok, model_input):
                continue

            idx = self.page.get_output_index(tok)
            bits = self.configuration.codebook.get_bits(idx)
            logit = L[bits].mean() if len(bits) > 1 else L[bits[0]]

            logits.append(TokenLogit(tok, float(logit)))

        logits.sort(key=lambda t: t.logit, reverse=True)
        return logits
    
    def select_token(self, token: Token, model_input: ModelInput) -> bool:
        if token.is_grammar() != model_input.grammar:
            return False
        if model_input.grammar:
            last = model_input.window.last_token()
            if last.text == token.text:
                return False
            if token.is_eol() and not last.is_punctuation():
                return False
            if token.is_punctuation() and last.is_terminal():
                return False
        return True

@dataclass
class PagedNetwork:
    configuration: Configuration
    activation_mlp: ActivationMLP
    page_list: list = field(default_factory=list)
    pages: dict = field(default_factory=dict)
    nets: dict = field(default_factory=dict)
        
    def _new_page(self, prev: str, target: Token) -> TokenPage:
        page = TokenPage(uuid4(), {prev}, [target])
        self.page_list.append(page)
        # deterministic, no network needed
        self.nets[page.uuid] = None
        return page

    def _create_default_network(self, page: TokenPage) -> PredictionNetwork:
        encoder: InputEncoder = InputEncoder()
        neural: NeuralNetwork = NeuralNetwork(
            self.configuration.generator_input_size(),
            self.configuration.hidden_size,
            self.configuration.output_dimension,
            self.activation_mlp)
        network: PredictionNetwork = PredictionNetwork(
            page,
            neural,
            encoder,
            self.configuration)
        self.nets[page.uuid] = network
        return network
        
    def learn_batch(self, batch: TrainingBatch):
        if len(batch.samples) == 0:
            return

        # cluster per page
        page_to_samples: dict[UUID, List[TrainingSample]] = {}

        for sample in batch.samples:
            page = sample.page
            if page.uuid not in page_to_samples:
                page_to_samples[page.uuid] = []
            page_to_samples[page.uuid].append(sample)

        # per page één batch update
        for page_uuid, samples in page_to_samples.items():
            net = self.nets[page_uuid]
            if net is not None:
                net.learn_batch(samples)
        
    def learn(self, model_input: ModelInput, target: Token, batch: TrainingBatch):
        prev: str = model_input.window.last_lemma_token()

        # Search page based on previous
        page: TokenPage = self.pages.get(prev, None)

        if page is None:
            # Find existing page with target token
            page: TokenPage = next((p for p in self.page_list if p.has_output_token(target)), None)

            if page:
                # Add to existing page
                page.add_input_token(prev)
            elif len(self.page_list) < self.configuration.total_pages:
                # Create a new page
                page: TokenPage = self._new_page(prev, target)
            else:
                # Find a page with smallest amount of outputs
                page: TokenPage = min(self.page_list, key=lambda p: p.input_size())
                
            # Register page for previous
            self.pages[prev] = page

        # Add target to page is not yet present
        if not page.has_output_token(target):
            page.add_output_token(target)
            
        # Obtain the network
        network = self.nets[page.uuid]
        if not network:
            network = self._create_default_network(page)            

        # Train the network
        network.learn_prebatch(model_input, target, batch)

    def propose(self, model_input: ModelInput):
        prev = model_input.window.last_lemma_token()

        # Determine page
        page: TokenPage = self.pages.get(prev, None)
        if page:
            return self.nets[page.uuid].propose(model_input)

        # Return empty result
        return []
