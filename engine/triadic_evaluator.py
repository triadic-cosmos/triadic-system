# triadic_evaluator.py
from dataclasses import dataclass, field
from os import listdir
from os.path import isfile, join
from typing import List

from .triadic_llm import TriadicLLM

MAX_TOKENS = 3000
MAX_STORIES = 1000
MAX_FILES = 1000

EVALUATION_PROMPT = (
"Give a score between 1 and 100 for the following short story. "
"The story is written on a moderate level. "
"A score below 50 is perfectly acceptable. "
"Assume the average moderate story scores around 50. "
"Only exceptional stories should score above 80. "
"Evaluate using the following criteria: " 
"content, coherence, readability, originality, creativity, style, structure. "
"Do not explain the score. Write the result as: Score: <number> and stop after that. "
"The story text is: $STORY"
)

@dataclass
class EvaluatorStory:
    sentences: List[str]    

@dataclass
class TriadicEvaluator:
    llm: TriadicLLM
    min_lines: int

    def score_story(self, story: EvaluatorStory) -> int:
        # score is minimum 1 and maximum 100
        score = 0
        while score <= 0 or score > 100:
            score = self.llm.score(story.sentences, "", MAX_TOKENS, EVALUATION_PROMPT)
        return score

    def evaluate_story(self, story: EvaluatorStory):
        # score story twice and store minumum and maximum
        score1 = self.score_story(story)
        score2 = self.score_story(story)
        story.min_score: int = min(score1, score2)
        story.max_score: int = max(score1, score2)
        story.score = (score1 + score2 + 1) // 2
        print(f"score = {story.min_score} / {story.max_score}")
        return
    
    def evaluate_file(self, filename: str):
        stories = []
        
        with open(filename, "r", encoding='utf-8-sig') as f:
            lines = f.read().splitlines()
            print(f"Processing {filename}")
            sentences = []
            
            for line in lines:
                if len(line) > 5:
                    sentences.append(line)
                else:
                    if len(sentences) >= self.min_lines:
                        stories.append(EvaluatorStory(sentences))
                        sentences = []
            
            if len(sentences) >= self.min_lines:
                stories.append(EvaluatorStory(sentences))
            
            print(f"stories = {len(stories)}")
            
            stories = stories[:MAX_STORIES]
            for story in stories:
                self.evaluate_story(story)

            min_scores = [story.min_score for story in stories]            
            max_scores = [story.max_score for story in stories]
            scores = [story.score for story in stories]

            self.output_file.write(f"{filename}\n")
            self.output_file.write(f"scores min = {min_scores}\n")
            self.output_file.write(f"scores max = {max_scores}\n")
            self.output_file.write(f"scores = {scores}\n")
            scores.sort()
            self.output_file.write(f"sorted = {scores}\n")
            self.output_file.write(f"total min = {sum(min_scores)}\n")                        
            self.output_file.write(f"total max = {sum(max_scores)}\n")
            self.output_file.write(f"total = {sum(scores)}\n")
            self.output_file.write(f"minimum = {min(scores)}\n")
            self.output_file.write(f"maximum = {max(scores)}\n")
            self.output_file.write(f"median = {scores[len(scores) // 2]}\n")
            self.output_file.flush()
    
    def evaluate_folder(self, folder: str, output_filename: str):
        with open(output_filename, "w", encoding="utf-8-sig") as self.output_file:
            files = [join(folder, f) for f in listdir(folder)]
            print(f"files = {len(files)}")
            files = files[:MAX_FILES]
            
            for file in files:
                if isfile(file):
                    self.evaluate_file(file)
