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

PROMPT = "Fix the following short story grammatically and semantically. " + \
    "Make it narratively coherent. Stay close to the original content. " + \
    "Avoid any duplication. End the story with a line containing: The End. " + \
    "This is the story: "
TITLE_PROMPT = "Create a nice chapter title for the following book chapter. " + \
    "Start with 'Chapter: ' followed by the chapter title. " + \
    "Stop after that. This is the chapter: "
TEMPERATURE = 0.9
TOP_P = 0.9
END_MARKER = "the end"

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

    def generate_title(self, lines: List[str], max_tokens: int) -> str:
        # Try with all lines
        prompt = TITLE_PROMPT + " ".join(lines)
        answer = self.generate(prompt, max_tokens)
        for line in answer.split("\n"):
            if line.startswith("Chapter:"):
                return line.replace("Chapter:", "").lstrip().rstrip()
        
        # Try with only first line
        prompt = TITLE_PROMPT + lines[0]
        answer = self.generate(prompt, max_tokens)
        for line in answer.split("\n"):
            if line.startswith("Chapter:"):
                return line.replace("Chapter:", "").lstrip().rstrip()
            
        # Fallback title
        return "Untitled"       
                
    def moderate(self, prefix: str, story: WriterStory, max_tokens: int) -> List[str]:
        prompt = PROMPT + story.get_story()
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

def clean_line(line:str) -> str:
    return line.lstrip().rstrip().replace("Dr.", "Dr").replace("Mr.", "Mr")
