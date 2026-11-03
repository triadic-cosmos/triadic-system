# writer_agent.py
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F

import random
import pickle
import math

from .config import Configuration
from .writer_environment import WriterEnvironment
from .writer_story import WriterStory, WriterSentence, WriterParams
from .tokens import Token, TokenLogit, TargetToken
from .context import ContextWindow, ModelInput
from .glp_network import GlpNetwork, Evaluation
from .training import TrainingSampleSentence
from .curriculum import Curriculum, CurriculumSentence
from .entity_tracker import TrackerMap
from .bias import BiasMLP, AMLPBias

GRAMMAR_CHECK = False

@dataclass
class WriterAgent:
    environment: WriterEnvironment
    id: str
    
    rng: random.Random = field(init=False)
    configuration: Configuration = field(init=False)
    glp_network: GlpNetwork = field(init=False)
    tracking: dict = field(init=False)

    training_count: int = 0
    training_epochs: int = 0

    def __init__(self, environment: WriterEnvironment, id: str):
        self.environment = environment
        self.configuration = self.environment.configuration
        self.id = id
        self.rng = random.Random()
        self.glp_network = GlpNetwork(self.configuration)

    def __str__(self):
        return f"[{self.id}] trainings = {self.training_count}"

    def new_context(self) -> ContextWindow:
        return ContextWindow(self.configuration,
                             self.glp_network.lemma_embedding_dict,
                             self.tracking)

    # ------------------------------------------------------------
    # Learning and curriculum training
    # ------------------------------------------------------------

    def add_tracking(self, curriculums: List[Curriculum]):
        if hasattr(self, "tracking") and self.tracking:
            print("Found existing tracking!")
            return
        tracking_map = TrackerMap(self.configuration)
        for curriculum in curriculums:
            tracking_map.add_entities(curriculum)
        tracking_map.partition_entities()
        self.tracking = tracking_map.partitions
        
    def add_bias(self, hidden_size, epochs: int):
        bias: BiasMLP = BiasMLP(self.configuration.other_hidden_size, hidden_size)
        bias.pretrain_sinus(epochs)
        self.glp_network.glp_network.bias = AMLPBias(bias)
        
    def plot_bias(self):
        if self.glp_network.glp_network.bias:
            self.glp_network.glp_network.bias.bias_mlp.plot()

    def train_curriculum(self, curriculum: Curriculum, random_epochs: int):
        # Train sentences using random order
        for epoch in range(1, random_epochs + 1):
            sentence: CurriculumSentence = curriculum.get_random_sentence(self.rng)
            self.glp_network.learn_sentence(sentence.sample)
            self.training_count += len(sentence.tokens)                
            if epoch % self.configuration.show_epochs_step == 0:
                print(epoch)

        # Show progress
        if epoch % self.configuration.show_epochs_step != 0:
            print(epoch)
        self.training_epochs += random_epochs
        self.show()

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------

    def evaluate_curriculum(self, curriculum: Curriculum):
        sentences = 0
        samples = 0
        allowed = 0
        both_blocked = 0
        grammar_blocked = 0
        lemma_blocked = 0
        top_matrix = 0
        top_mlp = 0
        top_combined = 0        
        total_matrix = 0 
        total_mlp = 0 
        total_combined = 0
        
        for index, sentence in enumerate(curriculum.sentences):
            sentences += 1
            evaluations: List[Evaluation] = self.glp_network.evaluate(sentence.sample)
            targets: List[str] = []
            
            for pos, evaluation in enumerate(evaluations):
                samples += 1
                if evaluation.allowed:
                    allowed += 1
                if evaluation.grammar_blocked and evaluation.lemma_blocked:
                    both_blocked += 1
                if evaluation.grammar_blocked:
                    grammar_blocked += 1
                if evaluation.lemma_blocked:
                    lemma_blocked += 1
                if evaluation.matrix >= 0:
                    top_matrix += 1
                if evaluation.mlp >= 0:
                    top_mlp += 1
                if evaluation.combined >= 0:
                    top_combined += 1
                total_matrix += evaluation.matrix
                total_mlp += evaluation.mlp
                total_combined += evaluation.combined
                if evaluation.allowed:
                    targets.append(evaluation.target.lemma.text)
                else:
                    targets.append(f"[{evaluation.target.lemma.text}]")
            
            print(f"[{index}] {' '.join(targets)}")

        print("=== Evaluation ===")
        print(f"sentences = {sentences}")
        print(f"samples = {samples}")
        print(f"allowed = {allowed}")
        print(f"both blocked = {both_blocked}")
        print(f"grammar blocked = {grammar_blocked}")
        print(f"lemma blocked = {lemma_blocked}")
        print(f"top matrix = {top_matrix}")
        print(f"top mlp = {top_mlp}")
        print(f"top combined = {top_combined}")
        print(f"total matrix = {total_matrix}")
        print(f"total mlp = {total_mlp}")
        print(f"total combined = {total_combined}")
  
    # ------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------

    def show(self, full = False):
        print(f"training epochs = {self.training_epochs}")
        print(f"training count = {self.training_count}")
        print(f"token input size = {self.configuration.generator_token_input_size()}")
        print(f"page count = {len(self.glp_network.page_list)}")
               
        if full:
            sizes = []
            for page in self.glp_network.page_list:
                sizes.append(page.get_size_text())
            print(sizes)

    def show_target(self):
        target_tokens = list(self.glp_network.get_target_tokens())
        print(f"total targets = {len(target_tokens)}")
        
        while len(target_tokens) > 0:
            first = target_tokens.pop(0)
            first_embedding = self.glp_network.lemma_embedding_dict.get_embedding(first).embedding
            first_tensor = torch.tensor(first_embedding, dtype=torch.float32)
            first_tensor = F.normalize(first_tensor, p=2, dim=0)
            similar_tokens = []
            
            for t in target_tokens:
                other_embedding = self.glp_network.lemma_embedding_dict.get_embedding(t).embedding
                other_tensor = torch.tensor(other_embedding, dtype=torch.float32)
                other_tensor = F.normalize(other_tensor, p=2, dim=0)
                score1 = float(torch.dot(first_tensor, other_tensor))
                score2 = torch.mean(torch.abs(other_tensor - first_tensor))
                if score1 >= 0.2 and score2 < 0.2:
                    similar_tokens.append(t)
                
            for t in similar_tokens:
                target_tokens.remove(t)
            
            print(f"{first.lemma} = {len(similar_tokens)}")

    def show_predictor(self):
        with torch.no_grad():
            m = self.glp_network.predictor_matrix
            row_norms = m.norm(dim=1)
            col_norms = m.norm(dim=0)

            print("=== Predictor Matrix Stats ===")            
            print(f"shape         : {tuple(m.shape)}")
            print(f"mean          : {m.mean().item():.6f}")
            print(f"std           : {m.std().item():.6f}")
            print(f"min           : {m.min().item():.6f}")
            print(f"max           : {m.max().item():.6f}")
            print(f"abs mean      : {m.abs().mean().item():.6f}")
            print(f"abs max       : {m.abs().max().item():.6f}")
            print(f"l2 norm       : {torch.norm(m).item():.6f}")
            print(f"p1            : {torch.quantile(m.flatten(), 0.01):.6f}")
            print(f"p5            : {torch.quantile(m.flatten(), 0.05):.6f}")
            print(f"p95           : {torch.quantile(m.flatten(), 0.95):.6f}")
            print(f"p99           : {torch.quantile(m.flatten(), 0.99):.6f}")
            print(f"row norm mean : {row_norms.mean():.6f}")
            print(f"row norm max  : {row_norms.max():.6f}")
            print(f"col norm mean : {col_norms.mean():.6f}")
            print(f"col norm max  : {col_norms.max():.6f}")            

    # ------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------

    def save(self, path: str):
        state = {
            "id": self.id,
            "config": self.configuration,
            "glp_network": self.glp_network,
            "training_count": self.training_count,
            "training_epochs": self.training_epochs,
            "tracking": self.tracking
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @staticmethod
    def load(environment: WriterEnvironment, path: str) -> "WriterAgent":
        with open(path, "rb") as f:
            state = pickle.load(f)

        agent = WriterAgent(environment, state["id"])
        agent.configuration = state["config"]
        agent.glp_network = state["glp_network"]
        agent.training_count = state["training_count"]
        agent.training_epochs = state["training_epochs"]
        agent.tracking = state["tracking"]
        
        if hasattr(agent.glp_network, "glp_network"):
            agent.glp_network.glp_network.initialize_bias()
        
        print(f"Loaded agent {agent.id}.")
        agent.show(True)
        return agent

    # ------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------

    def propose_token(self, params: WriterParams, model_input: ModelInput) -> Optional[TokenLogit]:
        outputs: List[TokenLogit] = self.glp_network.propose(params, model_input)

        if not outputs:
            return None

        # filtered output with blacklisted lemma list
        if params.lemma_blacklist:
            outputs = [o for o in outputs if o.lemma.text not in params.lemma_blacklist]
            if len(outputs) == 0:
                return None

        # top-k pairs
        top_k = params.top_k
        candidates = outputs[:top_k]

        # sampling using pair scores
        temperature = params.temperature
        logits = [c.logit for c in candidates]
        min_logit = min(logits)
        size = len(logits)
        boost_size = len(params.top_boost)
        
        for o in range(size):
            logits[o] = logits[o] - min_logit + temperature
            if o < boost_size:
                logits[o] = logits[o] * params.top_boost[o]
        total = sum(logits)
        if total == 0:
            probs = logits
        else:
            probs = [e / total for e in logits]
            
        selected = self.rng.choices(candidates, weights=probs, k=1)[0]

        # return full TokenLogit (grammar + lemma + logit)
        return selected

    def generate_sentence(
            self, params: WriterParams,
            model_input: ModelInput, sentences: List[str]) -> WriterSentence:
        generated_targets: List[TargetToken] = []
        generated: List[Token] = []
        ctx: ContextWindow = model_input.window

        for _ in range(params.max_tokens):
            proposal: TokenLogit = self.propose_token(params, model_input)
            if proposal is None:
                break

            # check if token is end of line
            if proposal.token.grammar.is_eol():
                generated.append(Token.EOL)
                
                # grammar validation on canonical tokens
                if self.environment.grammar.basic_validate_grammar_tokens(generated):
                    natural = self.environment.grammar.convert_from_canonical_tokens(generated)
            
                    # semantic validation
                    if self.environment.semantic.validate(params, sentences, natural):
                        return WriterSentence(generated_targets, generated, natural)

                # failed quality check
                return None

            generated_targets.append(proposal.token)
            ctx.add_target_token(proposal.token)
            generated.append(proposal.token.grammar)
            if proposal.token.lemma.is_lemma():
                generated.append(proposal.token.lemma)
            
        return None

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
    
    def update_context(self, ctx: ContextWindow, sentence: WriterSentence):
        ctx.start_sentence()
        ctx.update_entity_tracking(sentence.tokens)
        
        sentence_vector = self.glp_network.lemma_embedding_dict.get_sentence_embedding(sentence.target_tokens)
        ctx.update_sentence_memory(sentence_vector)

    def write_chapter(self, line_nr: int, max_lines: int, ctx: ContextWindow, params: WriterParams) -> WriterStory:
        chapter_lines = 0
        paragraph_lines = 0
        paragraph_divider = self.configuration.paragraph_divider
        fraction_multiplier: float = (params.to_line_fraction - params.from_line_fraction) / (params.lines - 1)
        sentences = []
        writer_sentences = []

        for _ in range(params.max_attempts):
            # Initialize model input
            ctx.clear_current_sentence()            
            line_position = [params.from_line_fraction + line_nr * fraction_multiplier, paragraph_lines / paragraph_divider]
            allow_paragraph = paragraph_lines >= params.min_lines_paragraph
            allow_chapter = chapter_lines >= params.min_lines_chapter and allow_paragraph
            model_input = ModelInput(ctx, line_position, allow_chapter, allow_paragraph)
            
            # Generate next sentence
            sentence = self.generate_sentence(params, model_input, sentences)
            if not sentence:
                continue
            if sentence.tokens[0].text == Token.SOC.text:
                # Next chapter is reached
                break            
            sentences.append(sentence.natural)
            writer_sentences.append(sentence)
            self.update_context(ctx, sentence)
            if len(sentences) >= max_lines:
                break

            # Update the line number
            line_nr += 1
            if sentence.tokens[0].text == Token.SOP.text:
                chapter_lines += 1
                paragraph_lines = 0
            else:
                chapter_lines += 1
                paragraph_lines += 1
            
        story = WriterStory(writer_sentences)
        self.fix_story(story)
        return story

    def write_story(
        self,
        prefix: str,
        ctx: ContextWindow,
        params: WriterParams
    ) -> WriterStory:
        
        chapters: int = 0
        line_nr: int = 0
        lines: int = params.lines
        chapter_lines = 0
        paragraph_lines = 0
        paragraph_divider = self.configuration.paragraph_divider
        fraction_multiplier: float = (params.to_line_fraction - params.from_line_fraction) / (lines - 1)

        # Prompt injection
        if params.prompt is not None and len(params.prompt) > 0:
            for prompt_line in params.prompt:
                raw_tokens = self.environment.grammar.convert_to_canonical_tokens(prompt_line)
                tokens = [self.token_dictionary.add_and_get(t.text) for t in raw_tokens]
                self.update_context_tokens(ctx, tokens)

        sentences = []
        writer_sentences = []

        print("> generating", end=" ")

        # Generate story line per line
        for _ in range(params.max_attempts):
            ctx.clear_current_sentence()

            line_position = [params.from_line_fraction + line_nr * fraction_multiplier, paragraph_lines / paragraph_divider]
            allow_paragraph = paragraph_lines >= params.min_lines_paragraph
            allow_chapter = chapter_lines >= params.min_lines_chapter and allow_paragraph
            model_input = ModelInput(ctx, line_position, allow_chapter, allow_paragraph)

            # Generate sentence
            sentence = self.generate_sentence(params, model_input, sentences)
            if not sentence:
                continue

            # Log progress
            print(f"{line_nr}", end=" ")

            # Update story
            sentences.append(sentence.natural)
            writer_sentences.append(sentence)

            # Update context window
            self.update_context(ctx, sentence)

            # Update line information
            line_nr += 1
            if line_nr == lines:
                break            
            if sentence.tokens[0].text == Token.SOC.text or \
               chapter_lines >= params.max_lines_chapter:
                chapter_lines = 0
                paragraph_lines = 0
                chapters += 1
                if line_nr + params.min_lines_chapter > lines or \
                   chapters >= params.max_chapters:
                    # Remove the new chapter
                    writer_sentences.pop()
                    break
                if sentence.tokens[0].text != Token.SOC.text:
                    sentence.target_tokens.insert(0, TargetToken(Token.SOC, Token.SOC))
                    sentence.tokens.insert(0, Token.SOC)
                    sentence.natural = "$ " + sentence.natural
            elif sentence.tokens[0].text == Token.SOP.text:
                chapter_lines += 1
                paragraph_lines = 0
            else:
                chapter_lines += 1
                paragraph_lines += 1

        print("done")

        # Build final story
        story = WriterStory(writer_sentences)

        # Grammar fix (canonical → natural)
        self.fix_story(story)

        print(f"{prefix}. {story.get_story()}")
        return story

    def build_output(self, output_path: str, params: WriterParams):
        index = 1

        with open(output_path, "w", encoding="utf-8-sig") as file:            
            while index <= params.amount:
                # Create new context
                ctx = self.new_context()
                if params.cas_vector:
                    print(f"Using CAS vector: {params.cas_vector}")
                    ctx.set_cas_vector(params.cas_vector)

                # Generate story
                story = self.write_story(
                    prefix=f"OUTPUT-{index}",
                    ctx=ctx,
                    params=params)

                # Write sentences to output
                file.write(f"========== OUTPUT {index} ==========\n")
                
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
