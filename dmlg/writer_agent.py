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
from .curriculum import Curriculum, CurriculumSentence

GRAMMAR_CHECK = False

@dataclass
class WriterAgent:
    environment: WriterEnvironment
    id: str
    
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
            self.glp_network.learn_batch(batch)
            self.training_count += len(batch.samples) 
            return TrainingBatch()
        else:
            return batch

    def train_curriculum(self, curriculum: Curriculum, random_epochs: int):
        super_batch: TrainingBatch = TrainingBatch()

        # Train sentences using random order
        for epoch in range(1, random_epochs + 1):
            sentence: CurriculumSentence = curriculum.get_random_sentence(self.rng)
            super_batch.append(sentence.batch)
            if epoch % self.configuration.epochs_step == 0:
                super_batch = self.learn_batch(epoch, super_batch)
            if epoch % self.configuration.show_epochs_step == 0:
                print(epoch)
        
        self.learn_batch(random_epochs + 1, super_batch)
        if epoch % self.configuration.show_epochs_step != 0:
            print(epoch)
        self.show()
        
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
        temperature = self.environment.configuration.beam_temperature
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
                    ModelInput(beam.ctx, model_input.line_position)
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
            # Applies some basic fixing rules to solve common issues
            fixed = sentence.natural            
            fixed = fixed.replace(" n't", " not")
            fixed = fixed.replace(" '", "'")
            fixed = fixed.replace(" a a", " an a")
            fixed = fixed.replace(" woulds", " would")            
            fixed = fixed.replace("I going", "I'm going")
            fixed = fixed.replace(" america", " America")
            sentence.fixed = fixed
            # Check with grammar checker if there are any other potential issues
            # Grammar fixing can be canon breaking and replace words incorrectly
            if GRAMMAR_CHECK:        
                grammar_fixed = self.environment.grammar.fix_grammar(fixed)
                if grammar_fixed != fixed:
                    print(f"<{fixed}")                
                    print(f">{grammar_fixed}")
    
    def update_context_tokens(self, ctx: ContextWindow, tokens: List[Token]):
        ctx.start_sentence()
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

        # Prompt injection
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

            line_position = [line_nr / (lines - 1)]
            model_input = ModelInput(ctx, line_position)

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

                # Write sentences to output
                file.write(f"========== BOOK {index} ==========\n")
                
                chapter = 1
                title = True
                for sentence in story.sentences:
                    line = sentence.fixed
                    if line[0] == '$':
                        line = line[2:]
                        title = True                        
                    elif line[0] == '#':
                        line = line[2:]                        
                        if not title:
                            file.write("\n")
                    if title:
                        file.write(f"\n   CHAPTER {chapter}.\n\n")     
                        title = False
                        chapter += 1
                    file.write(line + "\n")

                file.write("\n")
                index += 1
