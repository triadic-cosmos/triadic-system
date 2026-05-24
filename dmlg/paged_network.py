# paged_network.py
from typing import List
from dataclasses import dataclass, field
from uuid import uuid4, UUID
import torch
import random

from .config import Configuration
from .tokens import Token, TokenPage, TokenMapping, TokenLogit
from .context import ModelInput, InputEncoder
from .neural import NeuralNetwork
from .transition_map import TransitionMap
from .rule_based import RuleBasedFilter

rule_based = RuleBasedFilter()

@dataclass
class TrainingSample:
    page: TokenPage
    input_vector: List[float]
    target: Token

@dataclass
class TrainingBatch:
    samples: List[TrainingSample] = field(default_factory=list)

@dataclass
class PredictionNetwork:
    page: TokenPage
    input_size: int
    hidden_size: int
    activation_hidden_size: int
    encoder: InputEncoder
    configuration: Configuration
    network: NeuralNetwork = field(init=False)

    def __post_init__(self):
        self.network = NeuralNetwork(self.input_size, self.hidden_size, self.activation_hidden_size, self.page.output_size())

    def expand(self):
        self.network = self.network.add_output_channel()

    def learn(self, model_input: ModelInput, target: Token):
        x = torch.tensor([self.encoder.encode(model_input)], dtype=torch.float32)
        y = torch.tensor([self.page.to_output_mapping().encode(target)], dtype=torch.float32)
        opt = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        opt.zero_grad()
        pred = self.network(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        opt.step()

    def learn_prebatch(self, model_input: ModelInput, target: Token, batch: TrainingBatch):
        batch.samples.append(TrainingSample(self.page, self.encoder.encode(model_input), target))

    def learn_batch(self, samples: List[tuple]):
        xs = torch.tensor([x for (x, _) in samples], dtype=torch.float32)
        ys = torch.tensor([y for (_, y) in samples], dtype=torch.float32)

        opt = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        opt.zero_grad()

        pred = self.network(xs)
        loss = torch.nn.functional.mse_loss(pred, ys)

        loss.backward()
        opt.step()

        return loss.item()
        
    def propose(self, model_input: ModelInput, trans: List[Token], rng: random.Random):
        encoded_input = self.encoder.encode(model_input)
        x = torch.tensor([encoded_input], dtype=torch.float32)
        with torch.no_grad():
            out = self.network(x).squeeze(0).tolist()

        logits = []
        mapping = self.page.to_output_mapping()

        for pos, tok in mapping.position_to_token.items():
            logits.append(TokenLogit(tok, out[pos]))

        logits.sort(key=lambda t: t.logit, reverse=True)

        # --- Rule based logit filtering ---
        rule_filtered = rule_based.filter_logits(model_input, logits)

        # --- Transition-map top-logit filter ---
        best_tm_logit = None
        worst_tm_logit = None
        for tl in rule_filtered:
            if tl.token in trans:
                if best_tm_logit is None or tl.logit > best_tm_logit:
                    best_tm_logit = tl.logit
                if worst_tm_logit is None or tl.logit < worst_tm_logit:
                    worst_tm_logit = tl.logit
                    
        if best_tm_logit is None:
            best_tm_logit = logits[0].logit
            worst_tm_logit = best_tm_logit
                    
        upper_bound = best_tm_logit + self.configuration.score_upper_margin

        # allow more creativity at begin of a sentence
        if model_input.window.last_token().text == "<EOL>":
            span = best_tm_logit - worst_tm_logit
            lower_bound = best_tm_logit - rng.random() * span
        else:
            lower_bound = best_tm_logit - rng.random() * self.configuration.score_lower_margin    
            if lower_bound < worst_tm_logit:
                lower_bound = worst_tm_logit

        trans_filtered = [t for t in rule_filtered if t.logit >= lower_bound and t.logit <= upper_bound]
        return trans_filtered

@dataclass
class PagedNetwork:
    configuration: Configuration
    transition_map: TransitionMap
    page_list: list = field(default_factory=list)
    pages: dict = field(default_factory=dict)
    nets: dict = field(default_factory=dict)
    rng: random.Random = field(init=False)

    def __post_init__(self):
        self.rng = random.Random()

    def _new_page(self, prev: str, target: Token) -> TokenPage:
        page = TokenPage(uuid4(), {prev}, [target])
        self.page_list.append(page)
        # deterministic, no network needed
        self.nets[page.uuid] = None
        return page

    def _create_default_network(self, page: TokenPage) -> PredictionNetwork:
        encoder: InputEncoder = InputEncoder()
        network: PredictionNetwork = PredictionNetwork(
            page,
            self.configuration.generator_input_size(),
            self.configuration.hidden_size, # default hidden size
            0, # no learned activations by default
            encoder,
            Configuration)
        self.nets[page.uuid] = network
        return network

    def _create_optimized_network(self, page: TokenPage):
        input_size = page.input_size()
        output_size = page.output_size()
        hidden_size = suggest_hidden_size(input_size, output_size)
        activation_size = suggest_activation_hidden_size(input_size, output_size)
        if activation_size > 0:
            print(f"{input_size} -> {output_size} : {hidden_size} {activation_size}")
        # create optimized network
        encoder: InputEncoder = InputEncoder()
        network: PredictionNetwork = PredictionNetwork(
            page,
            self.configuration.generator_input_size(),
            hidden_size,
            activation_size,
            encoder,
            Configuration)
        self.nets[page.uuid] = network
        
    def learn_batch(self, batch: TrainingBatch):
        if len(batch.samples) == 0:
            return

        # 1. Cluster samples per page
        page_to_tuples = {}  # page.uuid -> list[(input_vector, output_vector)]

        for sample in batch.samples:
            page = sample.page
            mapping = page.to_output_mapping()
            output_vector = mapping.encode(sample.target)

            if page.uuid not in page_to_tuples:
                page_to_tuples[page.uuid] = []

            # Add tuple in the exact format your page-net expects
            page_to_tuples[page.uuid].append(
                (sample.input_vector, output_vector)
            )

        # 2. Per page één echte batch update
        for page_uuid, tuple_list in page_to_tuples.items():
            # tuple_list is now: [(input_vector, output_vector), ...]
            self.nets[page_uuid].learn_batch(tuple_list)
        
    def learn(self, model_input: ModelInput, target: Token, batch: TrainingBatch):
        prev: str = model_input.window.last_lemma_token()

        # 1. Transition map learning
        self.transition_map.learn(prev, target)

        # 2. Search page based on previous
        page: TokenPage = self.pages.get(prev, None)

        if page is None:
            # 3. Find existing page with target token
            page: TokenPage = next((p for p in self.page_list if p.has_output_token(target)), None)

            if page is not None and page.input_size() < self.configuration.max_page_inputs:
                # 4. Add to existing page
                page.add_input_token(prev)
            else:
                # 5. Create new page
                page: TokenPage = self._new_page(prev, target)
                
            # 6. Register page for previous
            self.pages[prev] = page

        # 7. Add target to page is not yet present
        if not page.has_output_token(target):
            if (page.is_deterministic()):
                page.add_output_token(target)
                # page is no longer deterministic
                self._create_default_network(page)
            else:
                page.add_output_token(target)
                # extend existing network
                self.nets[page.uuid].expand()

        # 8. Train the network
        if (page.is_deterministic()):
            return
        self.nets[page.uuid].learn_prebatch(model_input, target, batch)

    def propose(self, model_input: ModelInput):
        prev = model_input.window.last_lemma_token()

        # transition map
        trans = self.transition_map.get(prev)
        trans = rule_based.filter_tokens(model_input, trans)

        # exactly one transition → deterministic override
        if len(trans) == 1:
            return [TokenLogit(trans[0], 1.0)]

        # Fallback to paged network
        page: TokenPage = self.pages.get(prev, None)
        if page is None:
            return []

        # Deterministic page
        if page.is_deterministic():
            return [TokenLogit(page.output_tokens[0], 1.0)]            

        return self.nets[page.uuid].propose(model_input, trans, self.rng)
    
    def optimize(self):
        total_pages = len(self.page_list)
        total_networks = 0
        for page in self.page_list:
            if self.nets[page.uuid] != None:
                total_networks += 1
                self._create_optimized_network(page)
        print(f"total pages = {total_pages}")                
        print(f"total networks = {total_networks}")


def suggest_hidden_size(n_in, n_out, h_min=4, h_max=96):
    # geometric mean (for asymmetric mappings)
    base = (n_in * n_out) ** 0.5

    # boost if output is larger
    scale = 1.2 if n_out > n_in else 1.0

    # calculated hidden size
    h = int(base * scale)

    # clamp to min/max
    return max(h_min, min(h, h_max))

def suggest_activation_hidden_size(n_in, n_out):
    C = n_in * n_out
    if C < 100: return 0  # no activation network
    if C < 1000: return 4
    if C < 10000: return 6
    return 8
