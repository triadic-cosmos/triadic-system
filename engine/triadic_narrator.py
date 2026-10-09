# triadic_narrator.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import difflib
import time
import re

from .triadic_writer import TriadicWriter
from .triadic_llm import TriadicLLM

from dmlg import WriterParams, WriterStory

FIX_PROMPT = (
"Fix the following book chapter grammatically and semantically. "
"Make it narratively coherent, a literary masterpiece. " 
"Make sure there is proper and exciting narrative arc. "
"Try to stay close to the original content. "
"Keep all story elements related and realistic. "
"Fill the gaps and fix broken sentences. "
"No conversations, use purely descriptive sentences. Do not add a title! "
"Avoid any duplication. End the story with a line containing: The End. "
"This is the chapter: "
)

CHECK_PROMPT = (
"Check if the second chapter is a natural continuation of the first chapter. "
"Assume these should be two consecutive chapters in a real book. "
"Evaluate three dimensions internally: "
"1) Structural continuity (causality, time, objects, motivations), "
"2) Emotional continuity (tone, character consistency), "
"3) Thematic continuity (themes, atmosphere, narrative direction). "
"Combine these into a single score between 1 and 100, with 100 a perfect continuation. "
"Do not explain the score. Write the result as: Score: <number> and stop after that. "
"The first chapter has title '$TITLE1' and text: $SEQ1\n"
"The second chapter has title '$TITLE2' and text: $SEQ2"
)

BRIDGE_PROMPT = (
"Write a short transition of around 5 lines from the first chapter to the second chapter. "
"Keep it subtle and do not summarize or repeat either chapter. "
"Align the emotional, thematic and structural direction of the second chapter with the first chapter. "
"Do not introduce new events, characters or objects that are not implied by the chapters. "
"Avoid repetition, avoid recursive phrasing, avoid looping structures. "
"Start the transition with 'The Begin.' and end it with 'The End.' as markers. "
"Stop generation after writing the transition.\n"
"The first chapter has title '$TITLE1' and text: $SEQ1\n"
"The second chapter has title '$TITLE2' and text: $SEQ2"
)

# Narrator configuration
@dataclass(frozen=True)
class TriadicNarratorParams:
    input_filename: str
    output_filename: str
    max_chapters: int = 1000
    min_chapter_sentences: int = 15
    min_transition_sentences: int = 2
    max_transition_sentences: int = 15
    min_score: int = 85
    min_transition_score: int = 85
    max_retries: int = 15
    max_lines: int = 50
    max_tokens: int = 5000

# Narrator book chapter
@dataclass
class TriadicNarratorChapter:
    original_sentences: List[str]
     
# Narrator
@dataclass
class TriadicNarrator:
    llm: TriadicLLM
    params: TriadicNarratorParams

    def read_chapters(self) -> List[TriadicNarratorChapter]:
        with open(self.params.input_filename, "r", encoding="utf-8-sig") as input_file:
            lines = input_file.read().splitlines()
            lines.append("END CHAPTER")
        
            sentences = []
            chapters = []
            
            for line in lines:
                if line.startswith("="):
                    continue
                
                if "CHAPTER" in line:
                    if len(sentences) > 1:
                        sentences.pop(0)
                        sentences.pop(-1)
                        chapter = TriadicNarratorChapter(sentences.copy())
                        chapters.append(chapter)
                    sentences.clear()
                    continue
                
                sentences.append(line)
                    
            print(f"chapters = {len(chapters)}")
            return chapters
    
    def process_book(self):
        with open(self.params.output_filename, "w", encoding='utf-8-sig') as output_file:             
            start = time.perf_counter()

            print("Processing book")
            chapters = self.read_chapters()[:self.params.max_chapters]
            previous_chapter = None
            index = 0
            
            while index < len(chapters):
                chapter_nr = index + 1
                print(f"Processing chapter {chapter_nr}")
                chapter = chapters[index]
                 
                # Moderate chapter
                moderation_success = False
                chapter.moderated_sentences = chapter.original_sentences
                for t in range(self.params.max_retries):
                    moderated = self.llm.moderate(FIX_PROMPT, f"CHAPTER-{chapter_nr}",
                        chapter.original_sentences, self.params.max_tokens)
                    if len(moderated) >= self.params.min_chapter_sentences:
                        chapter.moderated_sentences = moderated
                        moderation_success = True
                        break
                                 
                # Determine a title for chapter
                for t in range(self.params.max_retries):
                    title = self.llm.generate_title(chapter.moderated_sentences, self.params.max_tokens)
                    title = title.replace('"', "")
                    chapter.title = title
                    if title != "Untitled":
                        break
                if chapter.title == "Untitled" and moderation_success:
                    continue # full retry of chapter when the moderation succeeded
                                                  
                # Determine a score for moderated chapter
                for t in range(self.params.max_retries):
                    score = self.llm.score(chapter.moderated_sentences, chapter.title, self.params.max_tokens)
                    chapter.moderated_score = score
                    if score > 0:
                        break
                if chapter.moderated_score < self.params.min_score and moderation_success:
                    continue # full retry of chapter when the moderation succeeded

                # Determine transition to previous chapter without retries
                transition = []
                if previous_chapter:                       
                    transition = bridge_chapters(self.llm, previous_chapter, chapter, self.params.max_tokens)
                    if len(transition) >= self.params.min_transition_sentences and \
                       len(transition) <= self.params.max_transition_sentences and \
                       transition[-1] != chapter.moderated_sentences[-1] and \
                        check_no_repetition(transition):
                        output_file.write("% TRANSITION\n")
                        for line in transition:
                            output_file.write(line + "\n")
                        output_file.write("\n")
                    previous_chapter.transition = transition
                previous_chapter = chapter
        
                # Write chapter to output file
                output_file.write(f"% {'-' * 50}\n")
                if moderation_success:
                    output_file.write("% CHAPTER")
                else:
                    output_file.write("% ORIGINAL CHAPTER")                
                output_file.write(f" {chapter_nr} : SCORE = {chapter.moderated_score}\n")
                output_file.write(f"% {'-' * 50}\n")
                output_file.write(f"\chapter{{{chapter.title}}}\n\n")
                for line in chapter.moderated_sentences:
                    output_file.write(line + "\n")
                output_file.write("\n")
                output_file.flush()
                
                # Go to next chapter
                index += 1 
                
            output_file.write(f"% Elapsed time : {time.perf_counter() - start:.1f} s\n\n")

# Iterative Narrator
@dataclass
class TriadicIterativeNarrator:
    llm: TriadicLLM
    writer: TriadicWriter
    params: TriadicNarratorParams
    writer_params: WriterParams

    def process_book(self):
        print("Processing iterative book")

        with open(self.params.output_filename, "w", encoding='utf-8-sig') as output_file:             
            start = time.perf_counter()

            ctx = self.writer.agent.new_context()
            chapter_nr = 1
            line_nr = 0
            previous_chapter = None
            
            while chapter_nr <= self.params.max_chapters and line_nr < self.writer_params.lines:
                print(f"Generating chapter {chapter_nr} at line {line_nr}")
                
                # Generate chapter
                copy_ctx = ctx.copy_full()
                story: WriterStory = self.writer.agent.write_chapter(
                    line_nr, self.params.max_lines, copy_ctx, self.writer_params)
                chapter = TriadicNarratorChapter(story.to_natural())
                print(f"Generated chapter has {len(chapter.original_sentences)} lines")                

                # Moderate chapter
                moderation_success = False
                for _ in range(self.params.max_retries):
                    moderated = self.llm.moderate(FIX_PROMPT, f"CHAPTER-{chapter_nr}",
                        chapter.original_sentences, self.params.max_tokens)
                    if len(moderated) >= self.params.min_chapter_sentences:
                        chapter.moderated_sentences = moderated
                        moderation_success = True
                        break                    
                if not moderation_success:
                    continue
                print(f"Moderated chapter has {len(chapter.moderated_sentences)} lines")                

                # Determine chapter title
                title_success = False
                for _ in range(self.params.max_retries):
                    title = self.llm.generate_title(chapter.moderated_sentences, self.params.max_tokens)
                    title = title.replace('"', "")
                    chapter.title = title
                    if title != "Untitled":
                        title_success = True
                        break
                if not title_success:
                    continue
                print(f"The chapter title is {chapter.title}")

                # Determine a score for moderated chapter
                chapter.moderated_score = 0
                for _ in range(self.params.max_retries):
                    score = self.llm.score(chapter.moderated_sentences, chapter.title, self.params.max_tokens)
                    chapter.moderated_score = score
                    if score > 0:
                        break
                if chapter.moderated_score < self.params.min_score:
                    continue
                print(f"The chapter has score {chapter.moderated_score}")

                # Determine a transition to previous chapter
                chapter.transition = None
                if previous_chapter:
                    for _ in range(self.params.max_retries):
                        transition = bridge_chapters(self.llm, previous_chapter, chapter, self.params.max_tokens)
                        if len(transition) >= self.params.min_transition_sentences and \
                           len(transition) <= self.params.max_transition_sentences and \
                           transition[-1] != chapter.moderated_sentences[-1] and \
                            check_no_repetition(transition):
                            # indicate transition by double empty line
                            chapter.transition = transition
                            previous_chapter.moderated_sentences.extend(transition)
                            print(f"The previous chapter has a transition of {len(transition)} lines")
                            break

                # Determine if the new chapter is a good transition from previous chapter
                chapter.transition_score = 100
                if previous_chapter:
                    check_prompt = bridge_prompt(CHECK_PROMPT, previous_chapter, chapter)
                    for _ in range(self.params.max_retries):
                        score = self.llm.score_prompt(check_prompt, self.params.max_tokens)
                        chapter.transition_score = score
                        if score > 0:
                            break
                if chapter.transition_score < self.params.min_transition_score:
                    continue
                print(f"The chapter has transition score {chapter.transition_score}")
        
                # Write chapter to output file
                if chapter.transition:
                    output_file.write("% TRANSITION\n")
                    for line in chapter.transition:
                        output_file.write(line + "\n")
                    output_file.write("\n")            
                output_file.write(f"% {'-' * 50}\n")
                output_file.write(f"% CHAPTER {chapter_nr} : SCORE = {chapter.moderated_score}\n")
                if previous_chapter:
                    output_file.write(f"% TRANSITION SCORE = {chapter.transition_score}\n")
                output_file.write(f"% {'-' * 50}\n")
                output_file.write(f"\chapter{{{chapter.title}}}\n\n")
                for line in chapter.moderated_sentences:
                    output_file.write(line + "\n")
                output_file.write("\n")
                output_file.flush()

                # Advance to the next chapter
                line_nr += len(chapter.original_sentences)
                previous_chapter = chapter                
                chapter_nr += 1
                ctx = copy_ctx

            output_file.write(f"% Elapsed time : {time.perf_counter() - start:.1f} s\n\n")

# ----- TRANSITION HELPERS -----

def bridge_prompt(prompt: str, first: TriadicNarratorChapter, second: TriadicNarratorChapter) -> str:
    return prompt \
        .replace("$TITLE1", first.title) \
        .replace("$SEQ1", " ".join(first.moderated_sentences)) \
        .replace("$TITLE2", second.title) \
        .replace("$SEQ2", " ".join(second.moderated_sentences))

def bridge_chapters(llm: TriadicLLM, first: TriadicNarratorChapter, second: TriadicNarratorChapter, max_tokens: int) -> List[str]:
    prompt = bridge_prompt(BRIDGE_PROMPT, first, second)
    transition = llm.generate(prompt, max_tokens)
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

def check_no_repetition(transitions: List[str],
                        prefix_len: int = 10,
                        min_repeat: int = 5,
                        similarity_threshold: float = 0.85) -> bool:
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

    return True
