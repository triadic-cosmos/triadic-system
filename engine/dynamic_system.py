# dynamic_system.py
from dataclasses import dataclass, field
from typing import List
import time

from .triadic_llm import TriadicLLM
from .triadic_writer import TriadicWriter

from dmlg import (
    GrammarEngine,
    SemanticEngine,
    WriterStory,
    WriterAgent,
    MultiAgent,
    Curriculum,
    CurriculumStory,
    CurriculumSentence,
    ContextWindow,
    WriterEnvironment,
    AgentBuilder,
    Configuration
)

# Dynamic parameters
@dataclass(frozen=True)
class DynamicParams:
    book: str
    curriculum: str
    stats: str
    exam: str
    exam_title: str
    hidden_size: int
    min_story_lines: int
    max_story_lines: int
    max_tokens: int
    min_score: int
    output_min_score: int
    exam_min_score: int
    train_epochs: int
    moderate_prompt: str
    score_prompt: str

# Dynamic system
@dataclass
class DynamicSystem:
    params: DynamicParams
    llm: TriadicLLM
    writer: TriadicWriter
    curriculum: Curriculum = field(default_factory = Curriculum)
    agent: WriterAgent = field(init=False)
    ctx: ContextWindow = field(init=False)

    def __post_init__(self):
        self.builder = self.writer.builder
        self.environment = self.writer.agent.environment
        self.configuration = self.environment.configuration
        self.ctx = ContextWindow(self.configuration)
        self.agent = self.writer.agent.agents[0]
        self.agent.configuration.hidden_size = self.params.hidden_size
        self.agent.configuration.story_lines = self.writer.num_lines

    def train_curriculum(self, epochs: int):
        self.agent.train_curriculum(self.curriculum, epochs, False)
        self.agent.build_index_from_curriculum(self.curriculum)

    def read_curriculum(self):
        curriculum_filename = self.writer.builder.curriculum_filename(
            self.environment, self.params.curriculum)
        self.curriculum.read_curriculum(curriculum_filename, self.environment)
        self.train_curriculum(self.params.train_epochs)

    def train_story(self, story: List[str]) -> CurriculumStory:
        doc = self.environment.grammar.nlp(" ".join(story))
        sentences = []
        for sent in doc.sents:
            canonical = self.environment.grammar.convert_to_canonical(sent.text)
            sentences.append(canonical)
        curriculum_story = self.curriculum.add_to_curriculum(sentences, self.environment)        
        self.train_curriculum(self.params.train_epochs)
        return curriculum_story

    def save(self):
        curriculum_filename = self.writer.builder.curriculum_filename(self.environment, self.params.curriculum)
        self.curriculum.write_curriculum_natural(curriculum_filename)
        self.agent.save(self.writer.builder.model_filename(self.environment))

    def add_to_context(self, story: CurriculumStory):
        if story:
            for s in story.sentences:
                sentence = self.environment.sentence_encoder.encode_sentence(s.tokens)
                self.ctx.add_sentence(sentence)

    def write_stats(self, stats_file, epoch: int, score: int, generated: int, published: int, reason: str):
        success = score >= self.params.min_score
        success_percent = generated * 100 / epoch
        training = self.agent.training_count
        pages = self.agent.nr_of_pages()
        page_ratio = self.agent.page_transition_ratio()
        stats_file.write(f"{epoch};{score};{success};{len(self.curriculum.stories)};{generated};{success_percent:.1f};{published};{training};{pages};{page_ratio:.1f};{reason}\n")
        stats_file.flush()

    def train(self, epochs: int, moderate: bool = True):
        self.read_curriculum()

        multi_agent: MultiAgent = MultiAgent(self.environment, [self.agent], [10], self.writer.variance)
        output_filename = self.builder.curriculum_filename(self.environment, self.params.book)
        stats_filename = self.builder.curriculum_filename(self.environment, self.params.stats)
        generated = 0
        published = 0

        with open(output_filename, "a", encoding='utf-8-sig') as file, \
             open(stats_filename, "a") as stats_file:
            for epoch in range(1, epochs + 1):
                print(f"epoch = {epoch}")
                
                # Create new story
                ctx_copy: ContextWindow = self.ctx.copy_current()
                story = multi_agent.write_story(f"GEN-{epoch}", ctx_copy)
                if len(story.sentences) < self.params.min_story_lines:
                    self.write_stats(stats_file, epoch, 0, generated, published, "DMLG")
                    continue

                # Moderate story
                if moderate:
                    moderated: List[str] = self.llm.moderate(self.params.moderate_prompt, f"LLM-{epoch}", story, self.params.max_tokens)
                    lines = len(moderated)                
                    if lines < self.params.min_story_lines or lines > self.params.max_story_lines:
                        self.write_stats(stats_file, epoch, 0, generated, published, "LMM")
                        continue
                else:
                    moderated: List[str] = [sentence.natural for sentence in story.sentences]

                # Generate a title for story
                title = self.llm.generate_title(moderated, self.params.max_tokens)
                if title == "Untitled":
                    self.write_stats(stats_file, epoch, 0, generated, published, "TITLE")
                    continue
                print(f"title = {title}")

                # Obtain the score for the generated story
                score = self.llm.score(moderated, title, self.params.max_tokens, self.params.score_prompt)
                if score < self.params.min_score:
                    self.write_stats(stats_file, epoch, score, generated, published, "SCORE")                    
                    continue
                print(f"score = {score}")

                # Publish story if it is good enough
                if score >= self.params.output_min_score:
                    published += 1
                    file.write(f"\\section{{{title}}}\n\n")
                    for line in moderated:
                        file.write(line + "\n")
                    file.write("\n\n")
                    file.flush()

                # Add story to curriculum and train agent
                curriculum_story = self.train_story(moderated)

                # Add story to the context
                self.add_to_context(curriculum_story)
                                
                # Save curriculum and agent
                self.save()

                # Show statistics
                ratio = self.agent.page_transition_ratio()
                generated += 1
                print(f"stories = {generated}, ratio = {ratio:.1f}")
                self.write_stats(stats_file, epoch, score, generated, published, "VALID")

    def examinate(self, amount: int):
        multi_agent: MultiAgent = MultiAgent(self.environment, [self.agent], [10], self.writer.variance)
        output_filename = self.builder.curriculum_filename(self.environment, self.params.exam)
        start = time.perf_counter()

        scores = []
        published = 0

        # --- Do the examination ---
        with open(output_filename, "w", encoding='utf-8-sig') as file:
            for index in range(1, amount + 1):
                story: WriterStory = multi_agent.write_story(f"GEN-{index}", self.ctx)
                lines = [sentence.natural for sentence in story.sentences]
                score = self.llm.score(lines, self.params.exam_title, self.params.max_tokens)
                scores.append(score)

                print(f"SCORE-{index} = {score}")

                # Write the best generated stories
                if score >= self.params.exam_min_score:
                    published += 1
                    for line in lines:
                        file.write(line + "\n")
                    file.write("\n\n")
                    file.flush()

            # --- Examination results ---
            if scores:
                scores_sorted = sorted(scores)
                avg_score = sum(scores) / len(scores)
                median_score = scores_sorted[len(scores)//2] if len(scores) % 2 == 1 else \
                    0.5 * (scores_sorted[len(scores)//2 - 1] + scores_sorted[len(scores)//2])
                min_score = scores_sorted[0]
                max_score = scores_sorted[-1]

                # Percentiles
                def percentile(p):
                    k = (len(scores_sorted) - 1) * p
                    f = int(k)
                    c = min(f + 1, len(scores_sorted) - 1)
                    return scores_sorted[f] + (scores_sorted[c] - scores_sorted[f]) * (k - f)

                p80 = percentile(0.80)
                p90 = percentile(0.90)

                file.write("=== Examination Statistics ===\n")
                file.write(f"Stories examined: {len(scores)}\n")
                file.write(f"Stories published (>={self.params.exam_min_score}): {published}\n")
                file.write(f"Average score: {avg_score:.2f}\n")
                file.write(f"Median score: {median_score:.2f}\n")
                file.write(f"Minimum score: {min_score:.2f}\n")
                file.write(f"Maximum score: {max_score:.2f}\n")
                file.write(f"80th percentile score: {p80:.2f}\n")
                file.write(f"90th percentile score: {p90:.2f}\n")
                file.write(f"Examination time: {time.perf_counter() - start:.1f} s\n")
