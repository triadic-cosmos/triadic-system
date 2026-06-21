# triadic_llm.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import List
import re

from dmlg import (
    WriterStory
)

# Model used for testing :
# https://huggingface.co/unsloth/mistral-7b-instruct-v0.3-bnb-4bit
MODEL_PATH = "../../mistral/"

TEMPERATURE = 0.9
TOP_P = 0.9
END_MARKER = "the end"

TITLE_PROMPT = \
    "Create a nice chapter title for the following book chapter. " + \
    "Start with 'Chapter: ' followed by the chapter title. " + \
    "Stop after that. This is the chapter: "
SCORE_PROMPT = \
    "Give a score between 0 and 100 for the following chapter and title. " + \
    "Evaluate using the following criteria: " + \
    "coherence, readability, originality, creativity, " + \
    "style, humor, narrative progression, suitability as a chapter in a real book." + \
    "Do not explain the score. Write the result as: Score: <number> and stop after that. " + \
    "The chapter title is: $TITLE\nThe chapter text is: $STORY"

@dataclass
class TriadicLLM:
    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")

    def generate(self, prompt: str, max_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P)        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def generate_title(self, lines: List[str], max_tokens: int, title_prompt: str = TITLE_PROMPT) -> str:
        # Try with all lines
        prompt = title_prompt + " ".join(lines)
        answer = self.generate(prompt, max_tokens)
        for line in answer.split("\n"):
            if line.startswith("Chapter:"):
                return line.replace("Chapter:", "").lstrip().rstrip()
        
        # Try with only first line
        prompt = title_prompt + lines[0]
        answer = self.generate(prompt, max_tokens)
        for line in answer.split("\n"):
            if line.startswith("Chapter:"):
                return line.replace("Chapter:", "").lstrip().rstrip()
            
        # Fallback title
        return "Untitled"       

    def validate(self, validate_prompt: str, story: str, max_tokens: int) -> bool:
        prompt = validate_prompt + story
        answer = self.generate(prompt, max_tokens)
        print(answer)
        for line in answer.split("\n"):
            lower_line = line.lower()
            if "my validation is: yes!" in lower_line:
                print("Validated!")
                return True
            if "my validation is: no!" in lower_line:
                print("Rejected!")
                return False
        return False

    def moderate(self, fix_prompt: str, prefix: str, story: WriterStory, max_tokens: int) -> List[str]:
        prompt = fix_prompt + story.get_story()
        answer = self.generate(prompt, max_tokens)
        print(answer)
        filtered = []
        
        for line in answer.split("\n"):
            line = clean_line(line)
            lower_line = line.lower()
            # remove output artifacts
            if len(lower_line) < 5 or \
               lower_line.startswith("fix the following") or \
               lower_line.endswith(":") or \
               "title:" in lower_line or \
               "corrected" in lower_line or \
               "revised" in lower_line:
                continue
            finished = False
            for output_line in re.split(r'(?<=[.!?])\s+', line):
                # check for end of story
                if END_MARKER in output_line.lower():
                    finished = True
                elif len(filtered) == 0 or filtered[-1] != output_line:
                        filtered.append(output_line)
            if finished:
                break
            
        print(f"{prefix}. {' '.join(filtered)}") 
        return filtered
    
    def score(self, lines: List[str], title: str, max_tokens: int, score_prompt: str = SCORE_PROMPT) -> int:
        prompt = score_prompt.replace("$TITLE", title).replace("$STORY", " ".join(lines))
        answer = self.generate(prompt, max_tokens)
        print(answer)
        for line in answer.split("\n"):
            score = parse_score(line)
            if score:
                return score
        return 0

def clean_line(line:str) -> str:
    return line.lstrip().rstrip().replace("Dr.", "Dr").replace("Mr.", "Mr")

def parse_score(text: str):
    match = re.search(r"Score:\s*(\d+)", text)
    if match:
        return int(match.group(1))
    return None
