# writer_agent.py
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional
from collections import defaultdict
import random
import pickle
import math

from .config import Configuration, TOP_BOOST
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

    def new_context(self) -> ContextWindow:
        return ContextWindow(self.configuration, self.glp_network.lemma_embedding_dict)

    # ------------------------------------------------------------
    # Learning and curriculum training
    # ------------------------------------------------------------

    def learn_batch(self, epoch: int, batch: TrainingBatch) -> TrainingBatch:
        if batch.has_samples():
            print(epoch)
            self.glp_network.learn_batch(batch)
            self.training_count += len(batch.samples) 
            return TrainingBatch()
        else:
            return batch

    def train_curriculum(self, curriculum: Curriculum, random_epochs: int):
        super_batch: TrainingBatch = TrainingBatch()

        # Train sequences using random order
        for epoch in range(1, random_epochs + 1):
            story: CurriculumStory = curriculum.get_random_story(self.rng)
            super_batch.append(story.batch)
            if epoch % self.configuration.epochs_step == 0:
                super_batch = self.learn_batch(epoch, super_batch)
        
        self.learn_batch(random_epochs + 1, super_batch)            
        self.show()
        
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

    def propose_token(self, model_input: ModelInput) -> Optional[TokenLogit]:
        outputs: List[TokenLogit] = self.glp_network.propose(model_input)

        if not outputs:
            return None

        # top-k pairs
        top_k = self.environment.configuration.top_k
        candidates = outputs[:top_k]

        # sampling using pair scores
        temperature = self.environment.configuration.temperature
        logits = [c.logit for c in candidates]
        min_logit = min(logits)
        size = len(logits)
        boost_size = len(TOP_BOOST)
        
        for o in range(size):
            logits[o] = logits[o] - min_logit + temperature
            if o < boost_size:
                logits[o] = logits[o] * TOP_BOOST[o]
        total = sum(logits)
        if total == 0:
            probs = logits
        else:
            probs = [e / total for e in logits]
            
        selected = self.rng.choices(candidates, weights=probs, k=1)[0]

        # return full TokenLogit (grammar + lemma + logit)
        return selected

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

    def generate_sentence(self, model_input: ModelInput, sentences: List[str]) -> WriterSentence:
        generated: List[Token] = []
        ctx: ContextWindow = model_input.window

        for _ in range(self.max_tokens):
            proposal: TokenLogit = self.propose_token(model_input)
            if proposal is None:
                break

            # --- 1. ALWAYS append grammar token ---
            grammar_tok = self.token_dictionary.add_and_get(proposal.grammar.text)
            generated.append(grammar_tok)
            ctx.add_token(grammar_tok)

            # --- 2. Append lemma token only if different from grammar ---
            if proposal.lemma is not None and proposal.lemma != proposal.grammar:
                lemma_tok = self.token_dictionary.add_and_get(proposal.lemma.text)
                generated.append(lemma_tok)
                ctx.add_token(lemma_tok)
            else:
                lemma_tok = grammar_tok   # terminal case

            # --- 3. EOL check MUST be on grammar token ---
            if grammar_tok.is_eol():
                # grammar validation on canonical tokens
                if self.environment.grammar.basic_validate_grammar_tokens(generated):
                    natural = self.environment.grammar.convert_from_canonical_tokens(generated)
            
                    # semantic validation
                    if self.environment.semantic.validate(sentences, natural):
                        return WriterSentence(generated, natural)

                # failed quality check
                return None

        # --- If no EOL was produced, close the line in context ---
        eol = self.token_dictionary.add_and_get(Token.EOL.text)
        ctx.add_token(eol)
        return None

    def generate_sentence_beam_search(
        self, model_input: ModelInput,
        keyword_scores: dict, used_tokens: set, sentences: List[str]
    ) -> WriterSentence:

        class Beam:
            def __init__(self, tokens: List[Token], ctx: ContextWindow, score: float):
                self.tokens = tokens
                self.ctx = ctx
                self.score = score
                self.eol = self.tokens[-1].is_eol() if self.tokens else False

        # --- config parameters ---
        temperature = self.environment.configuration.temperature
        alpha = self.environment.configuration.beam_alpha
        jitter_amp = self.environment.configuration.beam_jitter
        max_tokens = self.environment.configuration.max_tokens
        nr_of_beams = self.environment.configuration.nr_of_beams

        # --- initialize ---
        beams: List[Beam] = [Beam([], model_input.window.copy_current(), 0)]
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

                # GLP propose(): returns grammar+lemma pairs sorted by score
                outputs: List[TokenLogit] = self.glp_network.propose(
                    ModelInput(beam.ctx, model_input.sequence_embedding, model_input.line_number)
                )

                if not outputs:
                    continue

                # --- top-k pairs ---
                top_k = self.environment.configuration.top_k
                candidates = outputs[:top_k]

                # --- softmax over pair-scores ---
                logits = [c.logit for c in candidates]
                exp_logits = [math.exp(l / temperature) for l in logits]
                sum_exp = sum(exp_logits)
                softmax_scores = [e / sum_exp for e in exp_logits]

                # --- fork beams ---
                for idx, pair in enumerate(candidates):

                    # 1. Grammar token (always emitted first)
                    grammar_tok = self.token_dictionary.add_and_get(pair.grammar.text)

                    new_tokens = beam.tokens + [grammar_tok]
                    new_ctx = beam.ctx.copy_current()
                    new_ctx.add_token(grammar_tok)

                    # 2. Lemma token (only if different from grammar)
                    if pair.lemma != pair.grammar:
                        lemma_tok = self.token_dictionary.add_and_get(pair.lemma.text)
                        new_tokens.append(lemma_tok)
                        new_ctx.add_token(lemma_tok)

                    # 3. Score from softmax
                    score = softmax_scores[idx]

                    # 4. Damping
                    score = score ** alpha

                    # 5. Multipliers
                    if grammar_tok.is_eol():
                        score += 20.0
                    if grammar_tok.text not in used_tokens:
                        score *= 1.5

                    # 6. Jitter
                    score *= 1.0 + self.rng.random() * jitter_amp

                    # 7. Keyword bonus
                    score += keyword_scores.get(grammar_tok.text, 0)

                    # 8. Accumulate with beam history
                    total_score = score + beam.score * 0.9

                    # 9. Create new beam
                    new_beam = Beam(new_tokens, new_ctx, total_score)

                    # 10. Evaluate full sentence
                    if new_beam.eol:
                        if best_sentence is None or new_beam.score >= best_score:
                            # validate canonical tokens
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
    
    def update_context_tokens(self, ctx: ContextWindow, tokens: List[Token]):
        encoded = self.glp_network.sentence_encoder.encode_sentence(tokens)
        ctx.add_sentence(encoded)
        ctx.update_narrative_memory(tokens)
            
    def update_context(self, ctx: ContextWindow, sentence: WriterSentence):
        self.update_context_tokens(ctx, sentence.tokens)

    def write_story(
        self,
        prefix: str,
        ctx: ContextWindow,
        prompt: List[str] = None,
        keywords: Set[str] = None,
        beam_search: bool = False
    ) -> WriterStory:

        line_nr: int = 0
        lines: int = self.environment.configuration.story_lines

        # 1. Choose sequence embedding
        sequence_embedding = self.choose_best_embedding(keywords)

        # 2. Prompt injection
        if prompt is not None and len(prompt) > 0:
            for prompt_line in prompt:
                raw_tokens = self.environment.grammar.convert_to_canonical_tokens(prompt_line)
                tokens = [self.token_dictionary.add_and_get(t.text) for t in raw_tokens]
                self.update_context_tokens(ctx, tokens)

        sentences = []
        writer_sentences = []

        # Keyword scoring (lemma or terminal tokens)
        if keywords is not None:
            keyword_scores = {kw: 1.0 for kw in keywords}
            used_tokens = set()

        print("> generating", end=" ")

        beam_attempts = self.configuration.beam_attempts

        # 3. Generate story line per line
        for _ in range(self.environment.configuration.max_attempts):
            ctx.clear_current_sentence()

            line = [line_nr / self.configuration.line_divider]
            model_input = ModelInput(ctx, sequence_embedding, line)

            # --- BEAM SEARCH MODE ---
            if beam_search and beam_attempts > 0 and keywords is not None:
                sentence = self.generate_sentence_beam_search(
                    model_input,
                    keyword_scores,
                    used_tokens,
                    sentences
                )

                if not sentence:
                    beam_attempts -= 1
                    continue

                # update keyword usage
                for token in sentence.tokens:
                    used_tokens.add(token)
                    if token.text in keyword_scores:
                        keyword_scores[token.text] *= 0.9

            # --- NORMAL MODE ---
            else:
                sentence = self.generate_sentence(model_input, sentences)
                if not sentence:
                    continue

            # 4. Log progress
            print(f"{line_nr}", end=" ")

            # 5. Update story
            sentences.append(sentence.natural)
            writer_sentences.append(sentence)

            # 6. Update context window
            self.update_context(ctx, sentence)

            line_nr += 1
            if line_nr == lines:
                break

            beam_attempts = self.configuration.beam_attempts

        print("done")

        # 7. Build final story
        story = WriterStory(writer_sentences)

        # 8. Grammar fix (canonical → natural)
        self.fix_story(story)

        print(f"{prefix}. {story.get_story()}")
        return story

    def build_output(
        self,
        output_path: str,
        amount: int,
        prompt: List[str] = None,
        keywords: Set[str] = None,
        beam_search: bool = False
    ):
        index = 1

        with open(output_path, "w", encoding="utf-8-sig") as file:
            while index <= amount:
                # Create new context
                ctx = self.new_context()

                # Generate story
                story = self.write_story(
                    prefix=f"STORY-{index}",
                    ctx=ctx,
                    prompt=prompt,
                    keywords=keywords,
                    beam_search=beam_search
                )

                # Write fixed sentences to output
                for sentence in story.sentences:
                    file.write(sentence.fixed + "\n")

                file.write("\n")
                index += 1
