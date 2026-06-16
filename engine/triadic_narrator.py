# triadic_narrator.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import random
import re

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

DISTILL_PROMPT = "$CUSTOM " + \
"Start the story with 'The Begin.' and end it with 'The End.' as markers. " + \
"Stop generation after the story. "

VALIDATE_PROMPT = \
"Check the following story sequence if it has sufficient potential to fix into something really good. " + \
"Be extremely critical like a harsh human editor. $CUSTOM " + \
"Reject anything that is even slightly incoherent or weak. " + \
"Reject unless it is clearly fixable into a coherent story. " + \
"Reject if more than half of sentences are broken, repetitive, or semantically empty. " + \
"Reject if it is low quality, too random, broken or very incoherent. " + \
"Reply with 'My validation is: ' followed by 'NO!' when rejecting or by 'YES!' when accepting." 
"Stop after the validation. The sequence to check is: \n"
           
FIX_PROMPT = \
"Fix the following short story grammatically and semantically. " + \
"Make it narratively coherent. Stay close to the original content. $CUSTOM " + \
"No conversations, use purely descriptive sentences. Do not add a title! " + \
"Avoid any duplication. End the story with a line containing: The End. " + \
"This is the story: "

BRIDGE_PROMPT = \
"Write a short transition of about 3 lines from the first sequence to the second sequence. " + \
"Keep it subtle and do not summarize either sequence. $CUSTOM " + \
"Do not introduce new characters or objects that are not implied by the sequences. " + \
"Do not introduce new events, only create a mood‑based transition. " + \
"Start the transition with 'The Begin.' and end it with 'The End.' as markers. " + \
"Stop generation after writing the transition.\n" + \
"The first sequence has title '$TITLE1' and text: $SEQ1\n" + \
"The second sequence has title '$TITLE2' and text: $SEQ2"

KEYWORDS_PROMPT = \
"Determine a new set of up to 20 keywords for a sequence following given sequence. " + \
"Keywords can only be noun or verb lemmas. They should still fit the them of given sequence. " + \
"Avoid adding adverbs or adjectives. Do not conjugate the verbs, just give the base. " + \
"It is important that they allow a smooth and engaging narrative to continue. " + \
"Start the keyword list with 'Keywords: ' and give them as a comma-separated list. " + \
"Stop generation after giving the list. The titel for given sequence is: $TITLE\n" + \
"The keywords for the given sequence were : $KEYWORDS\nThe sequence is: "

END_PROMPT = \
"Check if the following story sequence would be a good ending for a book. " + \
"Reject the sequence if it feels like an early or premature ending. " + \
"Reject if major themes, conflicts, mysteries, or character arcs remain unresolved. " + \
"Reject if the sequence feels like the beginning or middle of a story. " + \
"Reject if the sequence lacks narrative closure, emotional resolution, or thematic completion. " + \
"Only accept if the sequence clearly provides a satisfying and intentional final chapter. " + \
"$CUSTOM" + \
"Reply with 'My validation is: ' followed by 'NO!' when this is not the case or by 'YES!' if this would be a good ending. " + \
"Stop after the validation. The title is: $TITLE\nThe sequence to check is: "

CURRICULUM = "book"
MAX_TOKENS = 2000
MIN_LINES = 7
MAX_LINES = 30
VARIANCE = 0.5
GEN_STORY_LINES = 15
MIX_STORY_LINES = 5
STORY_MIN_LINES = 10
BRIDGE_MIN_LINES = 2
RETRY_CHAPTERS = 1
ORIGINAL = False

# Narrator configuration
@dataclass(frozen=True)
class TriadicNarratorParams:
    name: str
    validate_custom: str
    fix_custom: str
    bridge_custom: str
    end_custom: str
    model: str
    prefixes: List[str]
    guest_models: List[str]
    guest_prefixes: List[str]
    keywords: set[str]
    min_score: int

# Dataset distiller
@dataclass
class TriadicDistiller:
    llm: TriadicLLM
    writer: TriadicWriter
    curriculum: Curriculum = field(default_factory = Curriculum)
    environment: WriterEnvironment = field(init=False)
    agent: WriterAgent = field(init=False)

    def __post_init__(self):
        self.environment = self.writer.agent.environment
        self.agent = self.writer.agent.agents[0]
        self.agent.configuration.hidden_size = 16
        self.agent.configuration.min_words = 3

    def generate_story(self, prompt: str) -> CurriculumStory:
        distill_prompt = DISTILL_PROMPT.replace("$CUSTOM", prompt)
        story = self.llm.generate(distill_prompt, MAX_TOKENS)
        story = story.replace("'", "").replace('"', "")
        started = 0
        sentences = []
        
        for output_line in re.split(r'(?<=[.!?])\s+', story):
            output_line = output_line.lstrip().rstrip()
            lower_line = output_line.lower()
            if "the begin" in lower_line:
                started += 1
            elif started == 2:
                if "the end" in lower_line:
                    break
                doc = self.environment.grammar.nlp(output_line)    
                for sent in doc.sents:
                    source = sent.text
                    canonical = self.environment.grammar.convert_to_canonical(source)
                    natural = self.environment.grammar.convert_from_canonical(canonical)
                    if len(natural) >= 5 and source.lower().startswith(natural.lower()):
                        sentences.append(canonical)
 
        if MIN_LINES <= len(sentences) <= MAX_LINES:
            return self.curriculum.add_to_curriculum(sentences, self.environment)
        return None

    def read_curriculum(self):
        curriculum_filename = self.writer.builder.curriculum_filename(self.environment, CURRICULUM)
        print(curriculum_filename)
        if Path(curriculum_filename).is_file():
            self.curriculum.read_curriculum(curriculum_filename, self.environment)

    def distill_llm(self, prompt: str, stories: int):
        generated = 0
        epoch = 0
        
        # Distillation loop        
        while True:
            epoch += 1
            print(f"Epoch {epoch}")
            story = self.generate_story(prompt)
            if story:
                # Train the agent once on every new curriculum story
                temp_curriculum = Curriculum()
                temp_curriculum.stories.append(story)
                self.agent.train_curriculum(temp_curriculum, 1, False)
                ratio = self.agent.page_transition_ratio()
                generated += 1
                print(f"stories = {generated}, ratio = {ratio}")                
                if generated >= stories:
                    break
        
        # Store the results
        curriculum_filename = self.writer.builder.curriculum_filename(self.environment, CURRICULUM)
        self.curriculum.write_curriculum_natural(curriculum_filename)
        self.agent.build_index_from_curriculum(self.curriculum)
        self.agent.save(self.writer.builder.model_filename(self.environment))

    def distill_dmlg(self, stories: int):
        generated = 0
        epoch = 0

        output_filename = self.writer.builder.curriculum_filename(self.writer.environment, CURRICULUM)
        with open(output_filename, "w", encoding='utf-8-sig') as file:
            while True:
                epoch += 1
                print(f"Epoch {epoch}")
                ctx: ContextWindow = ContextWindow(self.writer.configuration)
                story: WriterStory = self.writer.agent.write_story(f"STORY-{epoch}", ctx, None, None, False)
                self.writer.agent.fix_story(story)
                fix_prompt = FIX_PROMPT.replace("$CUSTOM", "")
                moderated = self.llm.moderate(fix_prompt, f"LLM-{epoch}", story, MAX_TOKENS)
                if MIN_LINES <= len(moderated) <= MAX_LINES:
                    for line in moderated:
                        file.write(line + "\n")
                    file.write("\n")
                    file.flush()
                    generated += 1
                    print(f"stories = {generated}")
                    if generated >= stories:
                        break

# Narrator book chapter
@dataclass
class TriadicNarratorChapter:
    title: str
    raw_story: WriterStory
    moderated_story: List[str]
    score: int

# Narrator
@dataclass
class TriadicNarrator:
    llm: TriadicLLM
    params: TriadicNarratorParams
    configuration: Configuration = field(init=False)
    environment: WriterEnvironment = field(init=False)
    builder: AgentBuilder = field(init=False)
    ctx: ContextWindow = field(init=False)
    rng: random.Random = field(init=False)
    
    def __post_init__(self):
        self.configuration: Configuration = Configuration(self.params.model)
        self.configuration.story_lines = GEN_STORY_LINES
        self.builder = AgentBuilder(self.configuration)
        self.ctx = ContextWindow(self.configuration)
        self.rng = random.Random()
        self.index = 0
        
        # Build the main writer ensemble
        self.agents = []
        for prefix in self.params.prefixes:
            agent_environment = self.builder.build_environment(self.configuration, prefix)
            agent: WriterAgent = self.builder.load_or_create_agent(agent_environment)
            self.agents.append(agent)
        
        # Build the guest writer ensemble
        self.guests = []
        for i in range(len(self.params.guest_models)):
            guest_config: Configuration = Configuration(self.params.guest_models[i])
            guest_environment: WriterEnvironment = self.builder.build_environment(guest_config, self.params.guest_prefixes[i])
            guest_agent: WriterAgent = self.builder.load_or_create_agent(guest_environment)
            self.guests.append(guest_agent)

        # Select environment
        if len(self.agents) > 0:
            self.environment = self.agents[0].environment        
        else:
            self.environment = self.guests[0].environment

    def bridge_chapters(self, first: TriadicNarratorChapter, second: TriadicNarratorChapter) -> List[str]:
        prompt = BRIDGE_PROMPT \
            .replace("$CUSTOM", self.params.bridge_custom) \
            .replace("$TITLE1", first.title) \
            .replace("$SEQ1", " ".join(first.moderated_story)) \
            .replace("$TITLE2", second.title) \
            .replace("$SEQ2", " ".join(second.moderated_story))
        
        transition = self.llm.generate(prompt, MAX_TOKENS)
        transition = transition.replace("'", "").replace('"', "")
        started = 0
        sentences = []
        
        for output_line in re.split(r'(?<=[.!?])\s+', transition):
            output_line = output_line.lstrip().rstrip()
            lower_line = output_line.lower()
            if "the begin" in lower_line:
                started += 1
            elif started == 2:
                if "the end" in lower_line:
                    break
                sentences.append(output_line)
        
        return sentences

    def create_ensemble_seed(self) -> WriterStory:
        # Generate a multi-agent
        ensemble = self.agents.copy()
        if len(self.guests) > 0:
            selected_guest = self.rng.choice(self.guests) 
            ensemble.append(selected_guest)
            print(f"guest = {selected_guest.configuration.name}")
        multi_agent: MultiAgent = MultiAgent(
            self.environment, ensemble, [10] * len(ensemble), VARIANCE)

        # Generate chapter seed with DMLG ensemble
        ctx_copy: ContextWindow = self.ctx.copy_current()
        story = multi_agent.write_story(f"GEN-{self.index}", ctx_copy)
                
        # Validate and moderate the generated sequence
        if not self.params.validate_custom:
            return story # override validation step
        validate_prompt = VALIDATE_PROMPT.replace("$CUSTOM", self.params.validate_custom)
        if self.llm.validate(validate_prompt, story.get_story(), MAX_TOKENS):
            return story
        return None

    def create_multi_seed(self) -> WriterStory:
        ctx_copy: ContextWindow = self.ctx.copy_current()
        multi_story: WriterStory = WriterStory([])
        for guest in self.guests:
            multi_agent: MultiAgent = MultiAgent(guest.environment, [guest], [10], VARIANCE)
            multi_agent.environment.configuration.story_lines = MIX_STORY_LINES
            story = multi_agent.write_story(f"GEN-{self.index}", ctx_copy)
            multi_story.sentences += story.sentences
        print(f"MULTI-{self.index}. {multi_story.get_story()}")
        return multi_story

    def add_to_context(self, moderated: List[str]):
        for line in moderated:
            tokens = self.environment.grammar.convert_to_canonical_tokens(line)
            sentence = self.environment.sentence_encoder.encode_sentence(tokens)
            self.ctx.add_sentence(sentence)

    def write_chapter(self) -> TriadicNarratorChapter:
        self.index += 1
        
        # Generate seed story with DMLG agents
        if len(self.agents) == 0:
            story = self.create_multi_seed()
        else:
            story = self.create_ensemble_seed()
        if not story:
            return None
        
        # Moderate generated seed story
        fix_prompt = FIX_PROMPT.replace("$CUSTOM", self.params.fix_custom)
        moderated: List[str] = self.llm.moderate(fix_prompt, f"LLM-{self.index}", story, MAX_TOKENS)
        if len(moderated) < STORY_MIN_LINES:
            return None
        # Generate a fitting title
        title = self.llm.generate_title(moderated, MAX_TOKENS)
        if title == "Untitled":
            return None
        
        # Obtain the score for the generated sequence
        score = self.llm.score(moderated, title, MAX_TOKENS)
        if score < self.params.min_score:
            return None
        print(f"Score {score} is accepted!")
        
        # Update the context with new story
        self.add_to_context(moderated)

        return TriadicNarratorChapter(title, story, moderated, score)

    def generate_chapters(self, nr_chapters: int) -> List[TriadicNarratorChapter]:
        chapters = []
        
        while len(chapters) < nr_chapters:
            chapter = self.write_chapter()
            if chapter:
                chapters.append(chapter)
                print(f"chapters = {len(chapters)}")
                
        chapters_sorted = sorted(chapters, key=lambda c: c.score, reverse=True)
        return chapters_sorted

    def output_book(self, chapters: List[TriadicNarratorChapter], nr_retries: int):
        output_filename = self.builder.curriculum_filename(self.environment, self.params.name)
        previous = None
        remaining = chapters
        failed = []
        retries = 0
        
        with open(output_filename, "w", encoding='utf-8-sig') as file:
            while True:
                # Check if a retry loop needs to be started
                if len(remaining) == 0:
                    if len(failed) > 0 and retries < nr_retries:
                        retries += 1
                        print(f"Starting retry loop {retries}!")
                        random.shuffle(failed)
                        # add additional chapters
                        remaining += self.generate_chapters(RETRY_CHAPTERS) 
                        remaining += failed
                        failed = []
                    else:
                        break
                
                # Take first remaining chapter
                chapter = remaining.pop(0)
                
                # Connect two chapters with a short transition as bridge
                if previous:
                    transition = self.bridge_chapters(previous, chapter)
                    if len(transition) <= BRIDGE_MIN_LINES:
                        print("Transition failed! " + chapter.title)
                        failed.append(chapter)
                        continue
                    file.write("\n")
                    for line in transition:
                        file.write(line + "\n")
                    file.write("\n\n")
                            
                # Write chapter title
                chapter_title = f"\\section{{{chapter.title}}}\n\n"
                file.write(chapter_title)
                
                if ORIGINAL:
                    # Write original story
                    file.write("\\begin{quote}\\small\n")
                    for sent in chapter.raw_story.sentences:
                        file.write(sent.natural + "\n")
                    file.write("\\end{quote}\n\n")
                    
                # Write fully moderated story
                for line in chapter.moderated_story:
                    file.write(line + "\n")
                
                # Whitespace
                file.flush()
                previous = chapter

    def determine_keywords(self, chapter: TriadicNarratorChapter, keywords: set[str]) -> set[str]:
        keywords_prompt = KEYWORDS_PROMPT.replace("$KEYWORDS", " ".join(keywords))
        keywords_prompt = keywords_prompt.replace("$TITLE", chapter.title)
        keywords_prompt += " ".join(chapter.moderated_story)
        answer = self.llm.generate(keywords_prompt, MAX_TOKENS)
        new_keywords: set[str] = extract_keywords(answer)
        if new_keywords:
            new_keywords = self.agents[0].filter_keywords(new_keywords)
            print(f"new keywords = {','.join(new_keywords)}\n")
            return new_keywords
        return keywords
                                                          
    def write_book(self, nr_chapters: int, nr_retries: int):        
        # Generate chapters
        chapters = self.generate_chapters(nr_chapters)

        # Output chapters
        self.output_book(chapters, nr_retries)

    def write_sequential_book(self, min_chapters: int, max_chapters: int, nr_retries: int):
        output_filename = self.builder.curriculum_filename(self.environment, self.params.name)
        multi_agent: MultiAgent = MultiAgent(self.environment, self.agents, [10] * len(self.agents), VARIANCE)
        fix_prompt = FIX_PROMPT.replace("$CUSTOM", self.params.fix_custom)
        end_prompt = END_PROMPT.replace("$CUSTOM", self.params.end_custom)
        previous: TriadicNarratorChapter = None
        keywords: set[str] = self.agents[0].filter_keywords(self.params.keywords)
        chapters: int = 0
        print(f"keywords = {','.join(keywords)}")
        
        with open(output_filename, "w", encoding='utf-8-sig') as file:
            while chapters < max_chapters:
                self.index += 1
                
                # Create seed story with keywords
                ctx_copy: ContextWindow = self.ctx.copy_current()
                story = multi_agent.write_story(f"GEN-{self.index}", ctx_copy, None, keywords, True)

                # Moderate seed story
                moderated: List[str] = self.llm.moderate(fix_prompt, f"LLM-{self.index}", story, MAX_TOKENS)
                if len(moderated) < STORY_MIN_LINES:
                    continue

                # Generate a fitting title
                title = self.llm.generate_title(moderated, MAX_TOKENS)
                if title == "Untitled":
                    continue

                # Obtain the score for the generated sequence
                score = self.llm.score(moderated, title, MAX_TOKENS)
                if score < self.params.min_score:
                    continue
                print(f"Score {score} is accepted!")

                # Create a chapter
                chapter = TriadicNarratorChapter(title, story, moderated, score)

                # Build a transition between two chapters
                if previous:
                    transition = None
                    for retry in range(1, nr_retries + 1):
                        transition = self.bridge_chapters(previous, chapter)
                        if len(transition) >= BRIDGE_MIN_LINES:
                            break
                        transition = None
                    if not transition:
                        print(f"Transition failed at try {retry}!")
                        continue                    
                    # Write transition
                    print(f"TRANSITION-{self.index}. {' '.join(transition)}")
                    file.write("\n")
                    for line in transition:
                        file.write(line + "\n")
                    file.write("\n\n")

                # Write new chapter
                chapter_title = f"\\section{{{chapter.title}}}\n\n"
                file.write(chapter_title)
                for line in chapter.moderated_story:
                    file.write(line + "\n")                
                file.flush()

                # Add new chapter to the context
                self.add_to_context(moderated)
 
                 # Set as new previous chapter
                previous = chapter
                chapters += 1
                print(f"chapters = {chapters}")
 
                # Check if this is a good chapter to end book
                if chapters >= min_chapters:              
                    full_end_prompt = end_prompt.replace("$TITLE", title)
                    if self.llm.validate(full_end_prompt, " ".join(moderated), MAX_TOKENS):
                        print("The end of book is reached!")
                        break
                    
                # Determine new the keyword set
                keywords = self.determine_keywords(chapter, keywords)


def extract_keywords(text: str) -> set[str] | None:
    """
    Extracts keywords from the *last* 'Keywords:' occurrence.
    Splits on commas first (primary separators), then splits each item on spaces.
    Returns a deduplicated set[str] of individual tokens.
    """
    marker = "Keywords:"
    idx = text.rfind(marker)  # last occurrence

    if idx == -1:
        return None

    # substring after last Keywords:
    part = text[idx + len(marker):].strip()
    if not part:
        return None

    # 1. split on commas → prevents full-string capture
    segments = part.split(",")

    keywords: set[str] = set()

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        # 2. split each segment on whitespace → individual tokens
        for token in seg.split():
            token = token.strip().lower()
            if token:
                keywords.add(token)

    return keywords if keywords else None
