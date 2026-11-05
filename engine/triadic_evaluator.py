# triadic_evaluator.py
from dataclasses import dataclass, field
from os import listdir
from os.path import isfile, join
from typing import List

from .triadic_llm import TriadicLLM

MAX_TOKENS = 1000
MAX_CHAPTERS = 1000
MIN_SENTENCES = 10

EVALUATION_PROMPT = (
"Give a score between 1 and 100 for the following book chapter. "
"It is a fiction book written on a moderate level. "
"A typical chapter scores around 50 and exceptional chapters can score above 80. "
"A score below 50 is perfectly acceptable. "
"Important are: readability, content, originality, coherence, structure. " 
"Do not explain the score. Write out the result formatted as: Score: <number>. "
"The chapter text is: $STORY"
)

@dataclass
class EvaluatorChapter:
    sentences: List[str]    

@dataclass
class TriadicEvaluator:
    llm: TriadicLLM

    def score_chapter(self, chapter: EvaluatorChapter) -> int:
        # score is minimum 1 and maximum 100
        score = 0
        while score <= 0 or score > 100:
            score = self.llm.score(chapter.sentences, "", MAX_TOKENS, EVALUATION_PROMPT)
        return score

    def evaluate_chapter(self, chapter: EvaluatorChapter):
        # score chapter twice and store minumum and maximum
        scores = []
        for _ in range(3):
            scores.append(self.score_chapter(chapter))
        scores.sort()            
        chapter.min_score: int = scores[0]
        chapter.max_score: int = scores[2]
        chapter.score: int = scores[1]
        print(f"scores = {scores}")
        return
    
    def evaluate_book(self, book_filename: str, evaluation_filename: str):
        chapters = []

        print(f"Processing {book_filename}")
        with open(book_filename, "r", encoding='utf-8-sig') as book_file:
            lines = book_file.read().splitlines()
            sentences = []
            
            for line in lines:
                if line.startswith("="):
                    continue
                if "CHAPTER" in line:
                    if len(sentences) > MIN_SENTENCES:
                        chapters.append(create_chapter(sentences))
                    sentences = []
                else:
                    sentences.append(line)
                                    
            if len(sentences) >= MIN_SENTENCES:
                chapters.append(create_chapter(sentences))
            
            print(f"chapters = {len(chapters)}")
            chapters = chapters[:MAX_CHAPTERS]

        with open(evaluation_filename, "w", encoding="utf-8-sig") as evaluation_file:
            chapter_nr: int = 1
            total_sentences = 0
            total_min_score = 0
            total_max_score = 0
            total_score = 0

            for chapter in chapters:
                self.evaluate_chapter(chapter)
                total_sentences += len(chapter.sentences)
                total_min_score += chapter.min_score
                total_max_score += chapter.max_score
                total_score += chapter.score                
                evaluation_file.write(f"{chapter_nr};{len(chapter.sentences)};{chapter.min_score};{chapter.max_score};{chapter.score}\n")
                evaluation_file.flush()
                chapter_nr += 1
            
            size = len(chapters)
            evaluation_file.write(f"AVG;{total_sentences/size:.1f};{total_min_score/size:.1f};{total_max_score/size:.1f};{total_score/size:.1f}\n")

def create_chapter(sentences: List[str]) -> EvaluatorChapter:
    if len(sentences[0]) < 5:
        sentences.pop(0)
    if len(sentences[-1]) < 5:
        sentences.pop()
    return EvaluatorChapter(sentences)
