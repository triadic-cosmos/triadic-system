# mapping_agent.py
from dataclasses import dataclass, field
from typing import List
import random
import re

from .writer_agent import WriterAgent, WriterEnvironment
from .context import ContextWindow, ModelInput
from .tokens import Token

MAX_LINES = 30
MIN_LENGTH = 3
MIN_SENTENCES = 3

@dataclass
class MappedSentence:
    tokens: List[Token]
    natural: str

@dataclass
class MappingAgent:
    environment: WriterEnvironment
    agent: WriterAgent
    rng: random.Random = field(init=False)

    def __post_init__(self):
        self.sentence_encoder = self.environment.sentence_encoder
        self.rng = random.Random()
        self.reset_sequence()
        self.current_line = 0

    def reset_sequence(self):
        self.sequence_embedding = self.agent.choose_best_embedding(None)

    # convert input text to tokens
    def convert_to_tokens(self, input_text: str) -> List[Token]:
        input_text = input_text.rstrip('\n')
        input_text = re.sub(r"[“”]", "", input_text)
        input_text = re.sub(r"[:;—]", ",", input_text)
        input_text = re.sub(r"[’]", "'", input_text)
        input_text = input_text.replace("-"," ")
        input_text = input_text.replace("_",",")
        canonical = self.environment.grammar.convert_to_canonical(input_text)
        canonical = canonical.replace("<SPLIT>", " <EOL>")
        tokens = [self.agent.token_dictionary.add_and_get(t) for t in canonical.split(" ") if t != ""]
        return tokens
        
    def map_sentence(self, ctx: ContextWindow,
                     mapped_sentences: List[MappedSentence],
                     current_sentence: List[Token],
                     input_text: str) -> bool:
        tokens = self.convert_to_tokens(input_text)
        if len(tokens) == 0:
            return True

        # map input tokens using a writer agent
        for token in tokens:
            # handle end of line token
            if token.is_eol():
                current_sentence.append(token)
                mapped_natural = self.environment.grammar.convert_from_canonical_tokens(current_sentence)
                if len(mapped_natural) >= MIN_LENGTH:
                    fixed_natural = self.environment.grammar.fix_grammar(mapped_natural)
                    fixed_natural = fixed_natural.lstrip().rstrip()
                    print(fixed_natural)
                    mapped_sentences.append(MappedSentence(current_sentence.copy(), fixed_natural))
                    ctx.add_sentence(self.sentence_encoder.encode_sentence(current_sentence))
                    ctx.update_narrative_memory(current_sentence)
                    self.current_line = (self.current_line + 1) % MAX_LINES
                else:
                    ctx.clear_current_sentence()
                current_sentence.clear()
                continue
            
            # reuse input grammar token
            if not token.is_lemma():
                current_sentence.append(token)
                ctx.add_token(token)
                continue
            
            # determine the next lemma token
            selected = token # use input token as default
            line_frac = self.current_line / (MAX_LINES - 1)
            model_input = ModelInput(ctx, self.sequence_embedding, line_frac)
            proposals = self.agent.paged_network.propose(model_input)
            lemma_proposals = [p.token for p in proposals if p.token.is_lemma()]
            if len(lemma_proposals) > 0:
                # Select the best candidate for mapping
                for proposal in lemma_proposals:
                    if proposal.text == token.text:
                        selected = proposal
                        break
                else:
                    selected = self.rng.choice(lemma_proposals[:3])                                
            current_sentence.append(selected)
            ctx.add_token(selected)
        
        return False

    def write_sequence(self, output_file, mapped_sentences: List[str]):
        if len(mapped_sentences) >= MIN_SENTENCES:
            for mapped_sentence in mapped_sentences:
                output_file.write(mapped_sentence.natural + "\n")
            output_file.write("\n")
            output_file.flush()
        mapped_sentences.clear()
        self.reset_sequence()
        
    def map_file(self, input_filename: str, output_filename: str):
        mapped_sentences = []
        current_sentence = []
        ctx: ContextWindow = ContextWindow(self.environment.configuration)

        with open(output_filename, "w", encoding='utf-8-sig') as output_file:
            with open(input_filename, "r", encoding="utf-8") as input_file:
                for line in input_file:
                    sequence = self.map_sentence(ctx, mapped_sentences, current_sentence, line)
                    if sequence:
                        self.write_sequence(output_file, mapped_sentences)
                self.write_sequence(output_file, mapped_sentences)
