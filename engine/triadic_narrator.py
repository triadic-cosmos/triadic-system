# triadic_narrator.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import difflib
import time
import re

from .triadic_llm import TriadicLLM

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

BRIDGE_PROMPT = \
"Write a short transition of about 3 lines from the first chapter to the second chapter. " + \
"Keep it subtle and do not summarize either chapter. " + \
"Do not introduce new characters or objects that are not implied by the chapters. " + \
"Do not introduce new events, only create a mood‑based transition. " + \
"Avoid repetition, avoid recursive phrasing, avoid looping structures. " + \
"Start the transition with 'The Begin.' and end it with 'The End.' as markers. " + \
"Stop generation after writing the transition.\n" + \
"The first chapter has title '$TITLE1' and text: $SEQ1\n" + \
"The second chapter has title '$TITLE2' and text: $SEQ2"

# Narrator configuration
@dataclass(frozen=True)
class TriadicNarratorParams:
    input_filename: str
    output_filename: str
    max_chapters: int = 1000
    min_chapter_sentences: int = 10
    min_transition_sentences: int = 2
    min_score: int = 85
    max_retries: int = 15
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
        
    def bridge_chapters(self, first: TriadicNarratorChapter, second: TriadicNarratorChapter) -> List[str]:
        prompt = BRIDGE_PROMPT \
            .replace("$TITLE1", first.title) \
            .replace("$SEQ1", " ".join(first.moderated_sentences)) \
            .replace("$TITLE2", second.title) \
            .replace("$SEQ2", " ".join(second.moderated_sentences))
        
        transition = self.llm.generate(prompt, self.params.max_tokens)
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
                    transition = self.bridge_chapters(previous_chapter, chapter)
                    if len(transition) >= self.params.min_transition_sentences and \
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
                 
# ----- HELPERS -----

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
