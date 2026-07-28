# triadic_narrator.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import difflib
import random
import re

from .triadic_llm import TriadicLLM
from .triadic_writer import TriadicWriter

from dmlg import (
    GrammarEngine,
    SemanticEngine,
    WriterStory,
    WriterSentence,
    WriterAgent,
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
"Avoid repetition, avoid recursive phrasing, avoid looping structures. " + \
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
MAX_TOKENS = 3000
MIN_LINES = 7
MAX_LINES = 30
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
    prefix: str
    keywords: set[str]
    min_score: int
    use_beam: bool

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
        self.agent = self.writer.agent
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
                self.agent.train_curriculum(temp_curriculum, 1, 0)
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
 
    def to_writer_story(self) -> WriterStory:
        story_sentences = []
        for sentence in self.moderated_story:
            story_sentences.append(WriterSentence([], sentence))
        return WriterStory(story_sentences)

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
        self.rng = random.Random()
        self.index = 0
        
        # Build the writer agent
        self.environment = self.builder.build_environment(self.configuration, self.params.prefix)
        self.agent: WriterAgent = self.builder.load_or_create_agent(self.environment)
        self.ctx = self.agent.new_context()

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

    def create_story(self) -> WriterStory:
        # Generate chapter seed with DMLG ensemble
        ctx_copy: ContextWindow = self.ctx.copy_current()
        story = self.agent.write_story(f"GEN-{self.index}", ctx_copy)
                
        # Validate and moderate the generated sequence
        if not self.params.validate_custom:
            return story # override validation step
        validate_prompt = VALIDATE_PROMPT.replace("$CUSTOM", self.params.validate_custom)
        if self.llm.validate(validate_prompt, story.get_story(), MAX_TOKENS):
            return story
        return None

    def add_to_context(self, moderated: List[str]):
        for line in moderated:
            tokens = self.environment.grammar.convert_to_canonical_tokens(line)
            sentence = self.agent.glp_network.sentence_encoder.encode_sentence(tokens)
            self.ctx.add_sentence(sentence)

    def write_chapter(self) -> TriadicNarratorChapter:
        self.index += 1
        
        # Generate seed story with DMLG agents
        story = self.create_story()
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
            new_keywords = self.agent.filter_keywords(new_keywords)
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
        fix_prompt = FIX_PROMPT.replace("$CUSTOM", self.params.fix_custom)
        end_prompt = END_PROMPT.replace("$CUSTOM", self.params.end_custom)
        previous: TriadicNarratorChapter = None
        keywords: set[str] = self.agent.filter_keywords(self.params.keywords)
        chapters: int = 0
        print(f"keywords = {','.join(keywords)}")
        
        with open(output_filename, "w", encoding='utf-8-sig') as file:
            while chapters < max_chapters:
                self.index += 1
                
                # Create seed story with keywords
                ctx_copy: ContextWindow = self.ctx.copy_current()
                story = self.agent.write_story(f"GEN-{self.index}", ctx_copy, None, keywords, self.params.use_beam)

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
                        if len(transition) >= BRIDGE_MIN_LINES and check_no_repetition(transition):
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

    def write_incremental_chapter(self, nr_candidates: int, nr_lines: int, keywords: set[str]) -> TriadicNarratorChapter:
        sequence_embedding = self.agent.choose_best_embedding(keywords)
        sentences: List[str] = []

        while len(sentences) < nr_lines:
            # Create candidate sentences
            candidates = []
            while len(candidates) < nr_candidates:
                sentence: WriterSentence = self.agent.generate_sentence(
                    sequence_embedding, self.ctx.copy_current(), sentences)
                if sentence and ends_with_punctuation(sentence.natural):                    
                    candidates.append(sentence.natural)
                    print(f"{len(candidates)}. {sentence.natural}")
            
            # Select best candidate
            best = candidates[0]
            for index in range(1, len(candidates)):
                if len(candidates[index]) > len(best):
                    best = candidates[index]
            
            # Add sentence to chapter
            fixed: str = self.environment.grammar.fix_grammar(best)            
            sentences.append(fixed)
            self.add_to_context(fixed)                
            print(f"lines = {len(sentences)} : {fixed}")
            
        return TriadicNarratorChapter("Short Story", None, sentences, 100)

    def write_remastered_chapter(self, chapter: TriadicNarratorChapter, retries: int) -> TriadicNarratorChapter:
        fix_prompt = FIX_PROMPT.replace("$CUSTOM", self.params.fix_custom)
        current_story: WriterStory = chapter.to_writer_story()
        current_score: int = self.params.min_score - 20
        empty_count: int = 0

        for index in range(1, retries + 1):
            # Remaster the combined generated chapters
            moderated: List[str] = self.llm.moderate(fix_prompt, f"REMASTER-{index}", current_story, MAX_TOKENS)
            if len(moderated) == 0:
                empty_count += 1
                if empty_count >= 3:
                    print("Moderation of chapter is aborted!")
                    return None
                continue
            
            # Quality check of remastered chapter
            if len(moderated) <= len(current_story.sentences) / 2 or \
               len(moderated) >= len(current_story.sentences) * 1.5 or \
               not check_no_repetition(moderated):
                continue
            
            # Determine title, this is mandatory                
            title = self.llm.generate_title(moderated, MAX_TOKENS)
            title = title.replace('"', "")                
            if title == "Untitled":
                continue

            # Determine score
            score = self.llm.score(moderated, title, MAX_TOKENS)
            print(f"Moderated chapter has score {score} and title {title}")
            chapter: TriadicNarratorChapter = TriadicNarratorChapter(title, None, moderated, score)
            if score >= self.params.min_score:        
                # Story is accepted as candidate
                return chapter
            if score >= current_score:
                # Use the better story as new moderation source
                current_story = chapter.to_writer_story()
                current_score = score
        
        # No remaster found within amount of tries with a high enough score
        print("Moderation of chapter has failed!")
        return None

    def write_incremental_book(self, nr_chapters: int, nr_candidates: int, nr_lines: int, retries: int, skip_intermediate: bool = True):
        total_chapters = 0
        output_filename = self.builder.curriculum_filename(self.environment, self.params.name)
        keywords: set[str] = self.agent.filter_keywords(self.params.keywords)
        print(f"keywords = {','.join(keywords)}")
        
        with open(output_filename, "a", encoding='utf-8-sig') as file:
            while True:
                # Reset context before each chapter
                self.ctx = self.agent.new_context()
                
                # Create incremental small chapters
                chapters: List[TriadicNarratorChapter] = []
                while True:
                    chapter: TriadicNarratorChapter = self.write_incremental_chapter(
                        nr_candidates, nr_lines, keywords)
                    chapters.append(chapter)
                    print(f"chapters = {len(chapters)}")
                    if len(chapters) >= nr_chapters:
                        break
                    keywords = self.determine_keywords(chapter, keywords)

                # Remaster incremental chapters and create bridges
                sentences = []
                previous_chapter = None
                remastered_chapters = []
                for chapter in chapters:
                    if skip_intermediate:
                        # no intermediate remastering
                        remastered_chapter = chapter 
                    else:
                        remastered_chapter = self.write_remastered_chapter(chapter, retries)
                    if remastered_chapter:
                        if previous_chapter and not skip_intermediate:
                            transition = self.bridge_chapters(previous_chapter, remastered_chapter)
                            if check_no_repetition(transition):
                                # reduce length of transition to maximum 10 lines
                                if len(transition) > 10:
                                    sentences += transition[:5]
                                    sentences += transition[-5:]
                                else:
                                    sentences += transition
                        sentences += remastered_chapter.moderated_story 
                        previous_chapter = remastered_chapter
                full_chapter: TriadicNarratorChapter = TriadicNarratorChapter( \
                    chapters[0].title, None, sentences, chapters[0].score)
                print(f"remastered lines = {len(sentences)}")
    
                # Create remastered full chapter
                remastered: TriadicNarratorChapter = self.write_remastered_chapter(full_chapter, retries)
                if not remastered:
                    continue

                # Write remastered full chapter
                file.write(f"\\section{{{remastered.title}}}\n\n")
                for line in remastered.moderated_story:
                    file.write(line + "\n")
                file.write("\n\n")
                file.flush()
                
                total_chapters += 1
                print(f">>> finished chapters = {total_chapters} <<<")


# ----- HELPERS -----
def ends_with_punctuation(line: str) -> bool:
    return line.rstrip().endswith(('.', '?', '!'))

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


def check_no_repetition(transitions: List[str],
                        prefix_len: int = 10,
                        min_repeat: int = 5,
                        similarity_threshold: float = 0.85,
                        structure_threshold: float = 0.75) -> bool:
    """
    Detects repetitive collapse in LLM transitions.
    Returns True if transitions are clean (no repetition), False if collapse detected.

    Parameters:
        transitions: list of generated lines
        prefix_len: number of characters to compare for prefix repetition
        min_repeat: number of consecutive lines needed to flag repetition
        similarity_threshold: semantic similarity threshold (0–1)
    """

    if len(transitions) < min_repeat:
        return True  # too short to detect collapse

    # --- 1. Check prefix repetition ---
    prefixes = [t[:prefix_len] for t in transitions]
    count = 1
    for i in range(1, len(prefixes)):
        if prefixes[i] == prefixes[i-1]:
            count += 1
            if count >= min_repeat:
                return False
        else:
            count = 1

    # --- 2. Check exact line repetition ---
    count = 1
    for i in range(1, len(transitions)):
        if transitions[i].strip() == transitions[i-1].strip():
            count += 1
            if count >= min_repeat:
                return False
        else:
            count = 1

    # --- 3. Check semantic similarity repetition ---
    # Uses difflib ratio to detect "same sentence with small changes"
    count = 1
    for i in range(1, len(transitions)):
        sim = difflib.SequenceMatcher(None, transitions[i], transitions[i-1]).ratio()
        if sim >= similarity_threshold:
            count += 1
            if count >= min_repeat:
                return False
        else:
            count = 1

    # --- 4. Check n-gram repetition (first 3 words) ---
    def first_words(line, n=3):
        return " ".join(line.split()[:n]).lower()

    ngrams = [first_words(t) for t in transitions]
    count = 1
    for i in range(1, len(ngrams)):
        if ngrams[i] == ngrams[i-1]:
            count += 1
            if count >= min_repeat:
                return False
        else:
            count = 1

    # --- 5. Structure-pattern repetition ---
    def skeleton(line: str) -> str:
        # remove names
        line = re.sub(r"\b(Jekyll|Hyde|Utterson)\b", "", line, flags=re.I)
        # remove nouns (approximation)
        line = re.sub(r"\b(man|beast|city|mirror|light|darkness|creation|soul)\b", "", line, flags=re.I)
        # collapse whitespace
        line = re.sub(r"\s+", " ", line).strip()
        return line.lower()

    skels = [skeleton(t) for t in transitions]

    count = 1
    for i in range(1, len(skels)):
        sim = difflib.SequenceMatcher(None, skels[i], skels[i-1]).ratio()
        if sim >= structure_threshold:
            count += 1
            if count >= min_repeat:
                return False
        else:
            count = 1

    return True
