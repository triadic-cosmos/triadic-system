# triadic_booster.py
from dataclasses import dataclass

from .triadic_llm import TriadicLLM

from dmlg import (
    AgentBuilder,
    Configuration,
    TrainingBatch,
    CurriculumSentence,
    CurriculumStory,
    ContextWindow,
    ModelInput,
    WriterAgent,
    WriterStory,
    WriterSentence
)

# Improve the story while keeping original canon
FIX_PROMPT = (
"Fix the following short story grammatically and semantically. "
"Fix broken sentences. Make it a literary masterpiece! "
"Avoid making it much longer than the original. "
"Avoid conversations, incoherence and repetition. "
"Avoid introducing a lot of new vocabulary. "
"Avoid excessive usage of punctuation and commas. "
"Use only narrative and descriptive story sentences in third person. "
"End the story with a line containing: The End. "
"This is the story: "
)

MAX_TOKENS = 3000
MAX_ATTEMPTS = 10

# Creates a booster curriculum from sampling an existing model
# This booster curriculum can be used to train or augment a model
@dataclass
class TriadicBooster:
    llm: TriadicLLM
    model_name: str
    model_prefix: str

    def __post_init__(self):
        self.configuration = Configuration(self.model_name)
        self.builder = AgentBuilder(self.configuration)
        self.environment = self.builder.build_environment(self.configuration, self.model_prefix)
        self.agent: WriterAgent = self.builder.load_or_create_agent(self.environment)
            
    def boost(self, output_filename, nr_stories: int, nr_lines: int, min_lines: int, max_lines: int, retries: int):
        with open(output_filename, "a", encoding='utf-8-sig') as file:
            generate_ctx: ContextWindow = self.agent.new_context()
            stories = 0
            epoch = 0
            
            while stories < nr_stories:
                epoch += 1
                sequence = self.agent.choose_best_embedding(None)
                sentences = []
                attempts = 0
                line_nr = 0
                
                while line_nr < nr_lines:
                    generate_ctx.clear_current_sentence()
                    line = [line_nr / self.configuration.line_divider]
                    model_input = ModelInput(generate_ctx, sequence, line)
                    sentence: WriterSentence = self.agent.generate_sentence(model_input, [])
                    if sentence:
                        sentence.fixed = self.environment.grammar.fix_grammar(sentence.natural)
                        # avoid stalling by using maximum number of attempts
                        if sentence.natural != sentence.fixed and attempts < MAX_ATTEMPTS:
                            attempts += 1
                            continue
                        attempts = 0
                        self.agent.update_context(generate_ctx, sentence)
                        sentences.append(sentence)
                        line_nr += 1
                        print(f"G-{len(sentences)}. {sentence.fixed}")

                writer_story: WriterStory = WriterStory(sentences)
                for _ in range(retries):
                    moderated: List[str] = self.llm.moderate(FIX_PROMPT, f"LLM-{epoch}", writer_story, MAX_TOKENS)
                    if len(moderated) >= min_lines:                        
                        cleaned = []
                        for sentence in moderated:
                            fixed = self.environment.grammar.fix_grammar(sentence)
                            if fixed == sentence:
                                cleaned.append(fixed)
                        if min_lines <= len(cleaned) <= max_lines:
                            break
                    cleaned = None
                if not cleaned:
                    continue

                index = 0
                for sentence in cleaned:
                    index += 1
                    file.write(sentence + "\n")    
                    print(f"M-{index}. {sentence}")
                file.write("\n\n")
                file.flush()
                stories += 1
                print(f"Finished story {stories}")
