# writer_agent.py
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional
from collections import defaultdict
import random
import pickle
import math

from .config import Configuration
from .writer_environment import WriterEnvironment
from .writer_story import WriterStory, WriterSentence
from .tokens import Token, TokenDictionary, TokenLogit
from .context import ContextWindow, ModelInput
from .glp_network import GlpNetwork, TrainingBatch
from .curriculum import Curriculum, CurriculumStory, CurriculumSentence

@dataclass
class WriterAgent:
    environment: WriterEnvironment
    id: str
    
    keyword_map: Dict[str, List[List[float]]] = field(init=False)
    keyword_count: Dict[str, int] = field(init=False)
    rng: random.Random = field(init=False)
    configuration: Configuration = field(init=False)
    glp_network: GlpNetwork = field(init=False)
    token_dictionary: TokenDictionary = field(init=False)

    training_count: int = 0

    def __init__(self, environment: WriterEnvironment, id: str):
        self.environment = environment
        self.configuration = self.environment.configuration
        self.id = id

        self.rng = random.Random()

        self.max_tokens = self.environment.configuration.max_tokens
        self.token_dictionary = TokenDictionary()
        self.keyword_map = dict()
        self.keyword_count = dict()

        self.glp_network = GlpNetwork(self.configuration, self.token_dictionary)

    def __str__(self):
        return f"[{self.id}] trainings = {self.training_count}"
        
    # ------------------------------------------------------------
    # Learning and curriculum training
    # ------------------------------------------------------------

    def train_story(self, epoch: int, story: CurriculumStory,
                    context_window: ContextWindow, batch: TrainingBatch) -> TrainingBatch:
        for sentence in story.sentences:
            # 1. train for each token
            for tok in sentence.tokens:
                target = self.token_dictionary.add_and_get(tok.text)
                model_input = ModelInput(context_window, story.embedding, target.is_grammar())
                self.training_count += 1
                self.glp_network.learn(model_input, target, batch)
                model_input.window.add_token(target)

            # 2. context / narrative for each sentence
            dict_tokens = [self.token_dictionary.add_and_get(t.text) for t in sentence.tokens]
            encoded = self.environment.sentence_encoder.encode_sentence(dict_tokens)
            context_window.add_sentence(encoded)
            context_window.update_narrative_memory(dict_tokens)

        if epoch % self.environment.configuration.epochs_step == 0:
            print(epoch)
            self.learn_batch(batch)
            return TrainingBatch()
        return batch

    def train_curriculum(self, curriculum: Curriculum, warmup_epochs: int, random_epochs: int):
        # Reuse context for all epochs
        context_window = ContextWindow(self.configuration)
        batch = TrainingBatch()

        # Train sequences using order curriculum 
        warmup_epoch = 1
        for _ in range(warmup_epochs):
            for story in curriculum.stories:
                batch = self.train_story(warmup_epoch, story, context_window, batch)
                warmup_epoch += 1

        self.learn_batch(batch)
        self.show()

        # New batch
        batch = TrainingBatch()
        
        # Train sequences using random order
        for epoch in range(1, random_epochs + 1):
            story = curriculum.get_random_story(self.rng)
            batch = self.train_story(epoch, story, context_window, batch)
        
        self.learn_batch(batch)
        self.show()

    def learn_batch(self, batch: TrainingBatch):
        if len(batch.samples) == 0:
            return
        self.glp_network.learn_batch(batch)

    # ------------------------------------------------------------
    # Curriculum story indexing
    # ------------------------------------------------------------

    def filter_keywords(self, keywords: set[str]) -> set[str]:
        filtered = set()
        for k in keywords:
            if k in self.keyword_map:
                filtered.add(k)
        return filtered

    def build_index_from_curriculum(self, curriculum: Curriculum):
        km = defaultdict(set) 
        kc = defaultdict(int)

        for story in curriculum.stories:
            emb = tuple(story.embedding)  
            kws = story.keywords

            for kw in kws:
                km[kw].add(emb)            
                kc[kw] += 1

        self.keyword_map = {k: list(v) for k, v in km.items()}
        self.keyword_count = dict(kc)

    def score_embeddings(self, keywords: Set[str]) -> Dict[tuple, float]:
        """
        Returns: embedding(tuple) -> score
        Score = sum( 1 / keyword_count[keyword] ) for all keywords the embedding belongs to.
        """
        scores = defaultdict(float)

        for kw in keywords:
            if kw not in self.keyword_map:
                continue
            weight = 1.0 / self.keyword_count[kw]
            
            for emb in self.keyword_map[kw]:
                emb_key = tuple(emb)
                scores[emb_key] += weight

        return scores

    def choose_best_embedding(self, keywords: Set[str]) -> List[float]:
        # ------------------------------------------------------------
        # 1. No keywords → random keyword → random embedding
        # ------------------------------------------------------------
        if not keywords:
            if not self.keyword_map:
                return None

            # random keyword
            kw = self.rng.choice(list(self.keyword_map.keys()))

            # random embedding for that keyword
            return list(self.rng.choice(self.keyword_map[kw]))

        # ------------------------------------------------------------
        # 2. Normal scoring using keywords
        # ------------------------------------------------------------
        scores = self.score_embeddings(keywords)
        if not scores:
            return None

        best_score = max(scores.values())

        # all embeddings with highest score
        candidates = [list(emb) for emb, sc in scores.items() if sc == best_score]

        # random choice from top scores
        return self.rng.choice(candidates)

    # ------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------

    def show(self, full = False):
        print(f"training count = {self.training_count}")
        print(f"page count = {len(self.glp_network.page_list)}")
        if full:
            sizes = []
            for page in self.glp_network.page_list:
                sizes.append(page.get_size_text())
            print(sizes)

    # ------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------

    def save(self, path: str):
        state = {
            "id": self.id,
            "config": self.configuration,
            "token_dictionary": self.token_dictionary,
            "glp_network": self.glp_network,
            "training_count": self.training_count,
            "keyword_map": self.keyword_map,
            "keyword_count": self.keyword_count,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @staticmethod
    def load(environment: WriterEnvironment, path: str) -> "WriterAgent":
        with open(path, "rb") as f:
            state = pickle.load(f)

        agent = WriterAgent(environment, state["id"])
        agent.configuration = state["config"]
        agent.token_dictionary = state["token_dictionary"]
        agent.glp_network = state["glp_network"]
        agent.training_count = state["training_count"]
        agent.keyword_map = state["keyword_map"]
        agent.keyword_count = state["keyword_count"]
        
        print(f"Loaded agent {agent.id}.")
        agent.show(True)
        return agent

    # ------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------

    def propose_token(self, model_input: ModelInput) -> Optional[Token]:
        outputs: List[TokenLogit] = self.glp_network.propose(model_input)

        if not outputs:
            return None

        # 1. Top-k selection
        top_k = self.environment.configuration.get_top_k(model_input.grammar)
        candidates = outputs[:top_k]

        # 2. Softmax with temperature
        temperature = self.environment.configuration.temperature 
        exps = [math.exp(t.logit / temperature) for t in candidates]
        total = sum(exps)
        probs = [e / total for e in exps]

        # 3. Probabilistic choice
        selected = self.rng.choices(candidates, weights=probs, k=1)[0]

        return selected.token

    def generate_keywords(self, prompt: str) -> set:
        all_keywords = set()
        for key in self.keyword_map.keys():
            all_keywords.add(key)
        keywords = set()
        tokens = self.environment.grammar.convert_to_canonical_tokens(prompt)
        for token in tokens:
            if token.text in all_keywords:
                keywords.add(token.text)
        return keywords
  
    def generate_sentence(self, sequence: List[float], ctx: ContextWindow, sentences: List[str]) -> WriterSentence:
        generated: List[Token] = []
        grammar = True

        for _ in range(self.max_tokens):
            proposal = self.propose_token(ModelInput(ctx, sequence, grammar))
            if proposal is None:
                break
            token = self.token_dictionary.add_and_get(proposal.text)
            generated.append(token)
            ctx.add_token(token)
            
            if proposal.is_eol():
                if self.environment.grammar.basic_validate_grammar_tokens(generated):
                    natural = self.environment.grammar.convert_from_canonical_tokens(generated)
                    if self.environment.semantic.validate(sentences, natural):        
                        return WriterSentence(generated, natural)
                return None # sentence failed quality check
            
            grammar = not grammar or proposal.is_terminal()

        # close unfinished line in context
        eol = self.token_dictionary.add_and_get(Token.EOL.text)
        ctx.add_token(eol)
        return None

    def generate_sentence_beam_search(
        self, sequence: List[float], ctx: ContextWindow,
        keyword_scores: dict, used_tokens: set, sentences: List[str]
    ) -> WriterSentence:

        class Beam:
            def __init__(self, tokens: List[Token], ctx: ContextWindow, score: float, grammar: bool):
                self.tokens = tokens
                self.ctx = ctx
                self.score = score
                self.grammar = grammar
                self.eol = self.tokens[-1].is_eol() if self.tokens else False

        # --- config parameters ---
        temperature = self.environment.configuration.temperature 
        alpha = self.environment.configuration.beam_alpha         
        jitter_amp = self.environment.configuration.beam_jitter
        max_tokens = self.environment.configuration.max_tokens
        nr_of_beams = self.environment.configuration.nr_of_beams

        # --- initialize ---
        beams: List[Beam] = [Beam([], ctx.copy_current(), 0, True)]
        best_sentence = None
        best_score = None

        # --- step over full token range ---
        for step in range(max_tokens):
            if len(beams) == 0:
                break

            new_beams: List[Beam] = []

            for beam in beams:
                if beam.eol:
                    new_beams.append(beam)
                    continue

                outputs: List[TokenLogit] = self.glp_network.propose(
                    ModelInput(beam.ctx, sequence, beam.grammar)
                )

                if not outputs:
                    continue

                # --- top-k selection ---
                top_k = self.environment.configuration.get_top_k(beam.grammar)
                candidates = outputs[:top_k]

                # --- softmax over logits ---
                logits = [c.logit for c in candidates]
                exp_logits = [math.exp(l / temperature) for l in logits]
                sum_exp = sum(exp_logits)
                softmax_scores = [e / sum_exp for e in exp_logits]

                for idx, logit in enumerate(candidates):
                    tok = logit.token
                    tok = self.token_dictionary.add_and_get(tok.text)

                    # fork context
                    new_ctx = beam.ctx.copy_current()
                    new_ctx.add_token(tok)
                    new_tokens = beam.tokens + [tok]

                    # softmax score
                    score = softmax_scores[idx]

                    # damping
                    score = score ** alpha

                    # multipliers
                    if tok.is_eol():
                        score *= 2.0
                    if tok.text not in used_tokens:
                        score *= 1.5

                    # jitter
                    score *= 1.0 + self.rng.random() * jitter_amp

                    # keyword bonus
                    score += keyword_scores.get(tok.text, 0)

                    # accumulate with beam history
                    total_score = score + beam.score * 0.9
                    new_grammar = not beam.grammar or tok.is_terminal()

                    # create new beam
                    new_beam = Beam(new_tokens, new_ctx, total_score, new_grammar)

                    # evaluate the new full sentence 
                    if new_beam.eol:
                        if best_sentence is None or new_beam.score >= best_score:
                            # validate canonical tokens (dictionary tokens zijn ok)
                            if not self.environment.grammar.basic_validate_grammar_tokens(new_tokens):
                                continue
                            natural = self.environment.grammar.convert_from_canonical_tokens(new_tokens)
                            if not self.environment.semantic.validate(sentences, natural):
                                continue
                            best_sentence = WriterSentence(new_tokens, natural)
                            best_score = new_beam.score

                    new_beams.append(new_beam)

            # prune beams
            new_beams.sort(key=lambda b: b.score, reverse=True)

            # beams = top_k best beams
            beams = new_beams[:nr_of_beams]

            if all(b.eol for b in beams):
                break

        if best_sentence:
            return best_sentence

        print("X", end="")
        return []

    def fix_story(self, story: WriterStory):
        for sentence in story.sentences:
            sentence.fixed = self.environment.grammar.fix_grammar(sentence.natural) 
    
    def write_story(self, prefix: str, ctx: ContextWindow, prompt: str = None, keywords: Set[str] = None, beam_search: bool = False) -> WriterStory:
        index: int = 0
        lines: int = self.environment.configuration.story_lines

        # use first agent to select sequence embedding
        sequence_embedding = self.choose_best_embedding(keywords)

        if prompt != None and len(prompt) > 0:
            while True:
                # fill up whole context with prompt
                for prompt_line in prompt:
                    raw_tokens = self.environment.grammar.convert_to_canonical_tokens(prompt_line)
                    tokens = [self.token_dictionary.add_and_get(t.text) for t in raw_tokens]
                    encoded = self.environment.sentence_encoder.encode_sentence(tokens)
                    ctx.add_sentence(encoded)
                    ctx.update_narrative_memory(tokens)
                if ctx.is_filled():
                    break
            
        sentences = []
        writer_sentences = []
        nr_of_tokens = self.environment.configuration.max_tokens
        if keywords != None:
            keyword_scores = dict()
            for keyword in keywords:
                keyword_scores[keyword] = 1.0
            used_tokens = set()           

        print("> generating", end= " ")
        beam_attempts = self.configuration.beam_attempts
        for i in range(self.environment.configuration.max_attempts):
            ctx.clear_current_sentence()
            
            if beam_search and beam_attempts > 0 and keywords != None:
                sentence = self.generate_sentence_beam_search(sequence_embedding, ctx, keyword_scores, used_tokens, sentences)
                if not sentence:
                    beam_attempts -= 1
                    continue
                for token in sentence.tokens:
                    used_tokens.add(token)
                    if token.text in keyword_scores:
                        keyword_scores[token.text] = keyword_scores[token.text] * 0.9                
            else:
                sentence = self.generate_sentence(sequence_embedding, ctx, sentences)
                if not sentence:
                    continue
        
            print(f"{index}", end = " ")
            sentences.append(sentence.natural)
            writer_sentences.append(sentence)
            encoded = self.environment.sentence_encoder.encode_sentence(sentence.tokens)
            ctx.add_sentence(encoded)
            ctx.update_narrative_memory(sentence.tokens)
            index += 1
            if index == lines:
                break
            beam_attempts = self.configuration.beam_attempts

        print("done")
        story = WriterStory(writer_sentences)
        self.fix_story(story)
        print(f"{prefix}. {story.get_story()}")
        return story

    def build_output(self, output_path: str, amount: int, prompt: List[str] = None, keywords: Set[str] = None, beam_search: bool = False):
        index = 1
        with open(output_path, "w", encoding='utf-8-sig') as file:
            while index <= amount:
                ctx: ContextWindow = ContextWindow(self.environment.configuration)            
                story = self.write_story(f"STORY-{index}", ctx, prompt, keywords, beam_search)
                for sentence in story.sentences:
                    file.write(sentence.fixed + "\n")
                file.write("\n")
                index += 1
