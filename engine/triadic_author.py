# triadic_author.py
from dataclasses import dataclass, field
from typing import List

from .triadic_writer import TriadicWriter
from .triadic_llm import TriadicLLM

from dmlg import (
    WriterStory,   
    WriterAgent,
    MultiAgent,
    WriterEnvironment,
    ContextWindow
)

MAX_TOKENS = 2000
MIN_LINES = 4
MAX_TRIES = 3

# Authoring engine
@dataclass
class TriadicAuthor:
    llm: TriadicLLM
    writers: List[TriadicWriter]
    environment: WriterEnvironment = field(init=False)
    multi_agent: MultiAgent = field(init=False)

    def __post_init__(self):
        agents: List[WriterAgent] = [writer.agent.agents[0] for writer in self.writers]
        self.environment = agents[0].environment
        self.multi_agent = MultiAgent(self.environment, agents, [1] * len(agents), self.writers[0].variance)

    def write_book(self, output_filename: str, title: str, chapters: int, lines: int):    
        ctx: ContextWindow = ContextWindow(self.environment.configuration)

        with open(output_filename, "w", encoding='utf-8-sig') as file:
            file.write(f"***** {title} *****\n\n")

            for chapter in range(1, chapters + 1):
                story: WriterStory = self.multi_agent.write_story(f"CHAPTER-{chapter}", ctx, None, None, False)
                for try_nr in range(1, MAX_TRIES + 1):               
                    moderated: List[str] = self.llm.moderate(f"LLM-{chapter}",story, MAX_TOKENS)
                    if len(moderated) < MIN_LINES:
                        continue
                    chapter_title = self.llm.generate_title(moderated, MAX_TOKENS)
                    if chapter_title == '':
                        continue
                    chapter_name = f"Chapter {chapter}. {chapter_title}"
                    file.write(f"*** {chapter_name} ***\n\n")
                    for line in moderated:
                        file.write(line + "\n")
                    file.write("\n")
                    file.flush()
                    print(f"Created {chapter_name} in try {try_nr}.")
                    break
