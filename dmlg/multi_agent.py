# multi_agent.py
from dataclasses import dataclass, field
from typing import Set, List, Optional
import random
import math

from .tokens import Token, TokenLogit
from .context import ContextWindow, ModelInput
from .writer_environment import WriterEnvironment
from .writer_agent import WriterAgent, WriterStory, WriterSentence

@dataclass
class MultiAgent:
    environment: WriterEnvironment
    agents: List[WriterAgent]
    weights: List[int]
    variance: float
    rng: random.Random = field(init=False)

    def __post_init__(self):
        self.sentence_encoder = self.environment.sentence_encoder
        self.rng = random.Random()
        self.beam_width = self.environment.configuration.nr_of_beams
        self.max_beam_width = self.environment.configuration.max_beams
        self.max_tokens = self.environment.configuration.max_tokens

    def select_agent(self) -> WriterAgent:
        selected: WriterAgent = None
        highest = -1
        for i in range(len(self.agents)):
            score = self.rng.random() * self.weights[i]
            if score > highest:
                highest = score
                selected = self.agents[i]
        return selected

    def propose_token(self, model_input: ModelInput, first: bool) -> Optional[Token]:
        for i in range(self.environment.configuration.token_retries):
            agent = self.select_agent()
            outputs: List[TokenLogit] = agent.paged_network.propose(model_input)

            if not outputs:
                continue

            if len(outputs) == 1:
                selected = outputs[0]
            elif first:
                selected = self.rng.choice(outputs)
            elif self.variance <= 0.0:
                selected = outputs[0]
            else:
                random_chance = self.rng.random()
                if random_chance <= self.variance:
                    selected = self.rng.choice(outputs)
                else:
                    # top 2 selection
                    selected = outputs[self.rng.randrange(0, 2)]
                
            return selected.token
        
        return None

    def generate_keywords(self, prompt: str) -> set:
        all_keywords = set()
        for agent in self.agents:
            for key in agent.keyword_map.keys():
                all_keywords.add(key)
        keywords = set()
        tokens = self.environment.grammar.convert_to_canonical_tokens(prompt)
        for token in tokens:
            if token.text in all_keywords:
                keywords.add(token.text)
        return keywords
            
    def generate_sentence(self, sequence: List[float], line: float, ctx: ContextWindow) -> List[Token]:
        generated: List[Token] = []
        first = True

        for _ in range(self.max_tokens):
            proposal = self.propose_token(ModelInput(ctx, sequence, line), first)
            if proposal is None:
                break
            generated.append(proposal)
            ctx.add_token(proposal)
            first = False
            if proposal == Token.EOL:
                return generated

        # close unfinished line in context
        ctx.add_token(Token.EOL)
        return []

    # beam search keyword driven generator
    def generate_sentence_beam_search(
        self, sequence: List[float], line: float, ctx: ContextWindow, keyword_scores: dict, used_tokens: set) -> List[Token]:

        class Beam:
            def __init__(self, tokens: List[Token], ctx: ContextWindow, score: float):
                self.tokens = tokens
                self.ctx = ctx
                self.score = score
                self.eol = self.tokens[-1].is_eol() if self.tokens else False

        # --- initialize beams ---
        beams: List[Beam] = [Beam([], ctx.copy_current(), 0)]

        for i in range(self.max_tokens):
            new_beams: List[Beam] = []

            for beam in beams:
                # stop expanding if already ended
                if beam.eol:
                    new_beams.append(beam)
                    continue

                # always propose next token using first agent
                agent = self.agents[0]
                outputs: List[TokenLogit] = agent.paged_network.propose(
                    ModelInput(beam.ctx, sequence, line)
                )

                if not outputs:
                    # dead beam, remove
                    continue

                # top-k selection (k = min(beam_width, len(outputs)))
                nr_outputs = len(outputs)
                k = min(self.beam_width, nr_outputs)

                top_logit = outputs[0].logit

                # collect logits for softmax
                logits = [l.logit for l in outputs]

                # softmax normalisation
                temperature = 0.3
                exp_logits = [math.exp(l / temperature) for l in logits]
                sum_exp = sum(exp_logits)

                for i in range(nr_outputs):
                    if i < k or self.rng.random() < self.variance:
                        logit = outputs[i]
                        tok = logit.token

                        # fork context
                        new_ctx = beam.ctx.copy_current()
                        new_ctx.add_token(tok)

                        new_tokens = beam.tokens + [tok]

                        # softmax score for token
                        softmax_score = math.exp(logit.logit / temperature) / sum_exp

                        # exponential damping
                        alpha = 0.75
                        softmax_score = softmax_score ** alpha

                        # jitter
                        jitter = self.rng.random() * 0.1

                        # total score
                        new_score = softmax_score + beam.score * 0.9 + jitter

                        # keyword bonus
                        new_score += keyword_scores.get(tok.text, 0)

                        # novelty bonus
                        if not logit.token.text in used_tokens:
                            new_score += 0.5

                        # EOL bonus
                        if tok.is_eol():
                            new_score += 10 * self.rng.random()

                        new_beam = Beam(new_tokens, new_ctx, new_score)
                        new_beams.append(new_beam)
            
            # prune to beam_width best beams
            new_beams.sort(key=lambda b: b.score, reverse=True)
            if i <= 3:
                beams = new_beams[:self.max_beam_width]
            else:
                beams = new_beams[:self.beam_width]

            # early stop if all beams ended
            if all(b.eol for b in beams):
                break

        if len(beams) == 0:
            return []

        # return best beam's tokens
        best = max(beams, key=lambda b: b.score)
        return best.tokens

    def evaluate_context(self, before_ctx: List[float], ctx: ContextWindow) -> float:
        if self.evaluator == None:
            return 1.0
        else:
            return self.evaluator.evaluate(before_ctx, ctx)

    def fix_story(self, story: WriterStory):
        for sentence in story.sentences:
            sentence.fixed = self.environment.grammar.fix_grammar(sentence.natural) 

    # writes single sentence using quality checks, adds sentence to context
    def write_sentence(self, sentences: List[str], ctx: ContextWindow, line: float, keywords: Set[str] = None) -> str:
        sequence = self.agents[0].choose_best_embedding(keywords)
        while True:
            sentence = self.generate_sentence(sequence, line, ctx)
            if self.environment.grammar.basic_validate_grammar_tokens(sentence):
                natural = self.environment.grammar.convert_from_canonical_tokens(sentence)
                if self.environment.semantic.validate(sentences, natural):
                    fixed = self.environment.grammar.fix_grammar(natural)
                    # use grammar fixed to check if there are no issues in sentence
                    if natural == fixed:
                        sentences.append(fixed)
                        encoded = self.sentence_encoder.encode_sentence(sentence)
                        ctx.add_sentence(encoded)
                        ctx.update_narrative_memory(sentence)
                        return fixed
    
    def write_story(self, prefix: str, ctx: ContextWindow, prompt: str = None, keywords: Set[str] = None, beam_search: bool = False) -> WriterStory:
        index:int = 0
        lines: int = self.environment.configuration.story_lines

        # use first agent to select sequence embedding
        sequence_embedding = self.agents[0].choose_best_embedding(keywords)

        if prompt != None and len(prompt) > 0:
            while True:
                # fill up whole context with prompt
                for prompt_line in prompt:
                    tokens = self.environment.grammar.convert_to_canonical_tokens(prompt_line)
                    encoded = self.sentence_encoder.encode_sentence(tokens)
                    ctx.add_sentence(encoded)
                    ctx.update_narrative_memory(tokens)
                if ctx.is_filled():
                    break
            
        sentences = []
        writer_sentences = []
        nr_of_beams = self.environment.configuration.nr_of_beams
        max_beams = self.environment.configuration.max_beams
        nr_of_tokens = self.environment.configuration.max_tokens
        if keywords != None:
            keyword_scores = dict()
            for keyword in keywords:
                keyword_scores[keyword] = 1.0
            used_tokens = set()           

        print("> generating", end= " ")
        for i in range(self.environment.configuration.max_attempts):
            ctx.clear_current_sentence()
            if lines == 1:
                line = 1
            else:
                line = index / (lines - 1)
            if beam_search and keywords != None:
                sentence = self.generate_sentence_beam_search(sequence_embedding, line, ctx, keyword_scores, used_tokens)
                for token in sentence:
                    used_tokens.add(token)
                    if token.text in keyword_scores:
                        keyword_scores[token.text] = keyword_scores[token.text] * 0.9                
            else:
                sentence = self.generate_sentence(sequence_embedding, line, ctx)
            
            if self.environment.grammar.basic_validate_grammar_tokens(sentence):
                natural = self.environment.grammar.convert_from_canonical_tokens(sentence)
                if self.environment.semantic.validate(sentences, natural):
                    print(f"{index}", end = " ")
                    sentences.append(natural)
                    writer_sentences.append(WriterSentence(sentence, natural, line))
                    encoded = self.sentence_encoder.encode_sentence(sentence)
                    ctx.add_sentence(encoded)
                    ctx.update_narrative_memory(sentence)
                    index += 1
                    if index == lines:
                        break

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
                fixed = self.fix_story(story)
                for sentence in story.sentences:
                    file.write(sentence.fixed + "\n")
                file.write("\n")
                index += 1
