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
from .bias import BiasMLP, AMLPBias

GRAMMAR_CHECK = False

@dataclass
class TrainingStats:
    def __init__(self):
        self.matrix_epochs: int = 0
        self.matrix_count: int = 0
        self.mlp_epochs: int = 0
        self.mlp_count: int = 0

    def show(self):
        print(f"matrix training epochs = {self.matrix_epochs}")
        print(f"matrix training samples = {self.matrix_count}")
        print(f"MLP training epochs = {self.mlp_epochs}")
        print(f"MLP training samples = {self.mlp_count}")

@dataclass
class ModelEvaluation:
    matrix_epochs: int
    mlp_epochs: int
    matrix_percent: float
    mlp_percent: float
    oracle_percent: float
    matrix_avg_reciprocal_rank: float
    mlp_avg_reciprocal_rank: float

    def is_matrix_improvement(self, previous: "ModelEvalation") -> bool:
        if not previous:
            return True
        return self.matrix_percent - previous.matrix_percent >= 0.1 or \
            self.matrix_avg_reciprocal_rank - previous.matrix_avg_reciprocal_rank >= 0.01

    def is_mlp_improvement(self, previous: "ModelEvaluation") -> bool:
        if not previous:
            return True
        return self.mlp_percent - previous.mlp_percent >= 0.1 or \
               self.mlp_avg_reciprocal_rank - previous.mlp_avg_reciprocal_rank >= 0.01

    def show(self):
        print(f"\n=== Evaluation ===")
        print(f"matrix epochs = {self.matrix_epochs}")
        print(f"MLP epochs = {self.mlp_epochs}")
        print("--- Average Reciprocal Rank ---")
        print(f"matrix = {self.matrix_avg_reciprocal_rank:.3f}")
        print(f"MLP = {self.mlp_avg_reciprocal_rank:.3f}")
        print("--- Top Rank % ---")
        print(f"matrix = {self.matrix_percent:.2f} %")
        print(f"MLP = {self.mlp_percent:.2f} %")
        print(f"oracle = {self.oracle_percent:.2f} %")
        print(f"oracle gap = {self.oracle_percent - self.mlp_percent:.2f} %")

@dataclass
class WriterAgent:
    environment: WriterEnvironment
    id: str
    
    rng: random.Random = field(init=False)
    configuration: Configuration = field(init=False)
    glp_network: GlpNetwork = field(init=False)
    tracking: dict = field(init=False)
    training: TrainingStats = field(init=False)

    def __init__(self, environment: WriterEnvironment, id: str):
        self.environment = environment
        self.configuration = self.environment.configuration
        self.id = id
        self.rng = random.Random()
        self.glp_network = GlpNetwork(self.configuration)
        self.tracking = None
        self.training = TrainingStats()

    def __str__(self):
        return f"[{self.id}]"

    def new_context(self) -> ContextWindow:
        return ContextWindow(self.configuration,
                             self.glp_network.lemma_embedding_dict,
                             self.tracking)

    # ------------------------------------------------------------
    # Learning and curriculum training
    # ------------------------------------------------------------
        
    def add_bias(self, hidden_size, epochs: int):
        bias: BiasMLP = BiasMLP(self.configuration.other_hidden_size, hidden_size)
        bias.pretrain_sinus(epochs)
        self.glp_network.glp_network.bias = AMLPBias(bias)
        
    def plot_bias(self):
        if self.glp_network.glp_network.bias:
            self.glp_network.glp_network.bias.bias_mlp.plot()
            
    def train_matrix(self, curriculum: Curriculum, random_epochs: int):
        # Train sentences using random order
        for epoch in range(1, random_epochs + 1):
            sentence: CurriculumSentence = curriculum.get_random_sentence(self.rng)
            self.glp_network.learn_sentence_matrix(sentence.sample)
            self.training.matrix_count += len(sentence.tokens)                
            if epoch % self.configuration.show_epochs_step == 0:
                print(epoch)

        self.training.matrix_epochs += random_epochs

    def train_mlp(self, curriculum: Curriculum, random_epochs: int):
        # Train sentences using random order
        for epoch in range(1, random_epochs + 1):
            sentence: CurriculumSentence = curriculum.get_random_sentence(self.rng)
            self.glp_network.learn_sentence_mlp(sentence.sample)
            self.training.mlp_count += len(sentence.tokens)                
            if epoch % self.configuration.show_epochs_step == 0:
                print(epoch)

        self.training.mlp_epochs += random_epochs
                        
    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------

    def evaluate_curriculum(self, curriculum: Curriculum, verbose: bool = False) -> ModelEvaluation:
        sentences = 0
        samples = 0
        allowed = 0
        both_blocked = 0
        grammar_blocked = 0
        lemma_blocked = 0
        top_both = 0
        top_only_matrix = 0
        top_only_mlp = 0
        top_none = 0
        total_matrix = 0 
        total_mlp = 0
        candidates = []
        agreements = []
        matrix_ranks = []
        mlp_ranks = []
        matrix_reciprocal = []
        mlp_reciprocal = []
        matrix_relative = []
        mlp_relative = []
        
        for index, sentence in enumerate(curriculum.sentences):
            sentences += 1
            evaluations: List[Evaluation] = self.glp_network.evaluate(sentence.sample)
            
            for pos, evaluation in enumerate(evaluations):
                samples += 1
                total_matrix += evaluation.matrix
                total_mlp += evaluation.mlp
                
                if evaluation.allowed:
                    allowed += 1
                if evaluation.grammar_blocked and evaluation.lemma_blocked:
                    both_blocked += 1
                if evaluation.grammar_blocked:
                    grammar_blocked += 1
                if evaluation.lemma_blocked:
                    lemma_blocked += 1
                    
                top_matrix = evaluation.rank_matrix <= 1
                top_mlp = evaluation.rank_mlp <= 1
                if top_matrix and top_mlp:
                    top_both += 1
                elif top_matrix:
                    top_only_matrix += 1
                elif top_mlp:
                    top_only_mlp += 1
                else:
                    top_none += 1
                
                candidates.append(evaluation.candidates)
                agreements.append(evaluation.agreement)
                matrix_ranks.append(evaluation.rank_matrix)
                mlp_ranks.append(evaluation.rank_mlp)
                matrix_reciprocal.append(1 / evaluation.rank_matrix)
                mlp_reciprocal.append(1 / evaluation.rank_mlp)
                matrix_relative.append(evaluation.rank_matrix / evaluation.candidates)
                mlp_relative.append(evaluation.rank_mlp / evaluation.candidates)
                            
        if samples <= 0:
            print("No samples to evaluate!")
            return
            
        candidates.sort()
        agreements.sort()
        matrix_ranks.sort()
        mlp_ranks.sort()
        matrix_reciprocal.sort()
        mlp_reciprocal.sort()
        matrix_relative.sort()
        mlp_relative.sort()
        
        median_index = samples // 2
        p10_index = samples // 10
        p90_index = (9 * samples) // 10

        matrix_avg_reciprocal = sum(matrix_reciprocal) / samples
        mlp_avg_reciprocal = sum(mlp_reciprocal) / samples

        matrix_percent = (top_both + top_only_matrix) * 100 / samples
        mlp_percent = (top_both + top_only_mlp) * 100 / samples
        oracle_percent = (top_both + top_only_matrix + top_only_mlp) * 100 / samples

        evaluation = ModelEvaluation(
            self.training.matrix_epochs, self.training.mlp_epochs,
            matrix_percent, mlp_percent, oracle_percent,
            matrix_avg_reciprocal, mlp_avg_reciprocal)

        if verbose:
            print(f"\n=== Evaluation ===")
            self.training.show()
            print("--- Curriculum ---")
            print(f"sentences = {sentences}")
            print(f"samples = {samples}")        
            print("--- Candidates ---")
            print(f"avg = {sum(candidates) / samples:.1f}")
            print(f"median = {candidates[median_index]}")
            print(f"max = {max(candidates)}")
            print("--- Rule Based ---")
            print(f"allowed = {allowed}")
            print(f"both blocked = {both_blocked}")
            print(f"grammar blocked = {grammar_blocked}")
            print(f"lemma blocked = {lemma_blocked}")
            print("--- Total Distance ---")
            print(f"matrix = {total_matrix:.5f}")
            print(f"MLP = {total_mlp:.5f}")
            print("--- Agreement ---")
            print(f"min = {min(agreements):.5f}")
            print(f"max = {max(agreements):.5f}")        
            print(f"avg = {sum(agreements) / samples:.5f}")
            print(f"median = {agreements[median_index]:.5f}")
            print(f"p10 = {agreements[p10_index]:.5f}")
            print(f"p90 = {agreements[p90_index]:.5f}")
            print("--- Rank Matrix ---")
            print(f"avg = {sum(matrix_ranks) / samples:.1f}")
            print(f"median = {matrix_ranks[median_index]}")
            print(f"reciprocal avg = {matrix_avg_reciprocal:.3f}")
            print(f"reciprocal median = {matrix_reciprocal[median_index]:.3f}")
            print(f"relative avg = {sum(matrix_relative) / samples:.3f}")
            print(f"relative median = {matrix_relative[median_index]:.3f}")
            print("--- Rank MLP ---")
            print(f"avg = {sum(mlp_ranks) / samples:.1f}")
            print(f"median = {mlp_ranks[median_index]}")
            print(f"reciprocal avg = {mlp_avg_reciprocal:.3f}")
            print(f"reciprocal median = {mlp_reciprocal[median_index]:.3f}")
            print(f"relative avg = {sum(mlp_relative) / samples:.3f}")
            print(f"relative median = {mlp_relative[median_index]:.3f}")
            print("--- Top Rank % ---")
            print(f"matrix = {matrix_percent:.2f} %")
            print(f"MLP = {mlp_percent:.2f} %")
            print(f"oracle = {oracle_percent:.2f} %")
            print(f"oracle gap = {oracle_percent - mlp_percent:.2f} %")
            print("--- Top Rank ---")        
            print(f"both = {top_both}")
            print(f"only matrix = {top_only_matrix}")
            print(f"only MLP = {top_only_mlp}")
            print(f"none = {top_none}")
        else:
            evaluation.show()
            
        return evaluation

    # ------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------

    def show(self, full = False):
        print(f"\n=== Statistics ===")
        self.training.show()        
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

    def show_matrix(self):
        with torch.no_grad():
            m = self.glp_network.predictor_matrix
            row_norms = m.norm(dim=1)
            col_norms = m.norm(dim=0)

            print("\n=== Predictor Matrix Stats ===")            
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
            "training": self.training,
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
        agent.training = state["training"]
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
            outputs = [o for o in outputs if o.token.lemma.text not in params.lemma_blacklist]
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
