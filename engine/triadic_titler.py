# triadic_titler.py
from dataclasses import dataclass
from typing import List

from .triadic_llm import TriadicLLM

MAX_TOKENS = 3000

@dataclass
class TriadicTitler:
    llm: TriadicLLM

    def title_book(self, book_filename: str, nr_titles: int):
        print(f"Processing {book_filename}")
        
        with open(book_filename, "r", encoding='utf-8-sig') as book_file:
            lines = book_file.read().splitlines()

            total_score = 0
            total_transition_score = 0
            titles = []
            book_titles = []

            for line in lines:
                if line.startswith("% CHAPTER"):
                    total_score += int(line.split('=')[1].strip())
                elif line.startswith("% TRANSITION SCORE"):
                    total_transition_score += int(line.split('=')[1].strip())
                elif line.startswith("\chapter{"):
                    start = line.index('{') + 1
                    end = line.rindex('}')
                    titles.append(line[start:end])

            print(titles)
            
            for index in range(1, nr_titles + 1):            
                title = self.llm.generate_book_title(titles, MAX_TOKENS)
                print(f"title {index} = {title}")
                book_titles.append(title)
            
            print(f"chapters = {len(titles)}")
            print(f"total score = {total_score}")
            print(f"total transition score = {total_transition_score}")
            print(f"average score = {total_score / len(titles):.1f}")
            print(f"average transition score = {total_transition_score / len(titles):.1f}")            
            for title in book_titles:
                print(title)
