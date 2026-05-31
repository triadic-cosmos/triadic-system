# moderated_agent.py
from dataclasses import dataclass, field
from typing import List
import random

from .tokens import Token, TokenLogit
from .context import ContextWindow, ModelInput
from .writer_environment import WriterEnvironment
from .writer_agent import WriterAgent, WriterStory, WriterSentence
from .sentence_encoder import SentenceEncoder

# Moderated generated sentence
@dataclass
class ModeratedSentence:
    tokens: List[Token]
    votes: List[int]
    success: bool
    
    def confidence(self, moderators: int) -> float:
        total_votes = 0
        total_lemmas = 0
        for index in range(len(self.tokens)):
            if self.tokens[index].is_lemma():
                total_lemmas += 1              
                total_votes += self.votes[index]
        return total_votes / (total_lemmas * moderators)

# Writer agent moderated by ensemble of other agents
@dataclass
class ModeratedAgent:
    environment: WriterEnvironment
    writer: WriterAgent
    moderators: List[WriterAgent]
    rng: random.Random = field(init=False)
    sentence_encoder: SentenceEncoder = field(init=False)
    nr_moderators: int = field(init=False)

    def __post_init__(self):
        self.rng = random.Random()
        self.sentence_encoder = SentenceEncoder()
        self.nr_moderators = len(self.moderators)

    def propose_token(self, agent: WriterAgent, model_input: ModelInput, first: bool) -> List[TokenLogit]:
        outputs: List[TokenLogit] = agent.paged_network.propose(model_input)
        if outputs:
            return outputs
        return []

    def generate_sentence(self, sequence: List[float], line: float, ctx: ContextWindow, min_grammar_votes: int) -> ModeratedSentence:
        max_tokens: int = self.environment.configuration.max_tokens
        generated_tokens: List[Token] = []
        # number of moderators that agree on each token
        generated_votes: List[int] = []
        success = False

        for pos in range(max_tokens):
            first = pos == 0
            model_input = ModelInput(ctx, sequence, line)
            # Get proposal from writer agent
            proposal = self.propose_token(self.writer, model_input, first)
            # No proposal
            if not proposal:
                break
            votes = dict()
            for token in proposal:
                votes[token.token] = 0
            
            # Get proposals from moderator agents
            for moderator in self.moderators:
                moderator_proposal = self.propose_token(moderator, model_input, first)
                # Validator has no proposal
                if not moderator_proposal:
                    continue
                for p in moderator_proposal:
                    if p.token in votes.keys():
                        votes[p.token] += 1
            
            max_vote = max(votes.values())
            if max_vote == 0:
                # when no votes always take best token
                selected = proposal[0].token 
            else:
                best_proposals = [t for t in votes.keys() if filter_token(votes[t], max_vote, min_grammar_votes, t.is_lemma())]
                # grammar tokens need vote from a minumum number of moderators
                if not best_proposals:
                    break
                selected = self.rng.choice(best_proposals)
                
            generated_tokens.append(selected)
            generated_votes.append(max_vote)
            ctx.add_token(selected)
            
            if selected == Token.EOL:
                success = True
                break

        # close the unfinished line in context
        if not success:
            ctx.add_token(Token.EOL)
        return ModeratedSentence(generated_tokens, generated_votes, success)

    def generate_stories(self, output_filename: str, stories: int, lines: int, min_grammar_votes: int, min_confidence: float = 0):
        with open(output_filename, "w", encoding='utf-8-sig') as file:
            for story in range(1, stories + 1):
                print(f"Generating story {story}")
                ctx: ContextWindow = ContextWindow(self.environment.configuration)
                line = 1
                sequence = self.writer.choose_best_embedding(None)
                sentences = []
                
                while line <= lines:
                    line_fraction = (line - 1) / (lines - 1)
                    moderated_sentence = self.generate_sentence(sequence, line_fraction, ctx, min_grammar_votes)
                    if moderated_sentence.success:
                        confidence = moderated_sentence.confidence(self.nr_moderators)
                        if confidence >= min_confidence:
                            sentence = moderated_sentence.tokens
                            if self.environment.grammar.basic_validate_grammar_tokens(sentence):
                                natural = self.environment.grammar.convert_from_canonical_tokens(sentence)
                                if self.environment.semantic.validate(sentences, natural):
                                    fixed = self.environment.grammar.fix_grammar(natural)
                                    # check if there are no grammatical issues
                                    if natural == fixed:
                                        marked = mark_sentence(fixed, moderated_sentence, self.nr_moderators)
                                        sentences.append(fixed)
                                        encoded = self.sentence_encoder.encode_sentence(sentence)
                                        ctx.add_sentence(encoded)
                                        text = f"{line}. [{confidence:.2f}] {marked}"
                                        print(text)
                                        file.write(text + "\n")
                                        line += 1
                
                file.write("\n")
                file.flush()

# Filter a token based on number of votes
def filter_token(votes: int, max_votes: int, min_grammar_votes: int, lemma: bool):
    if lemma or max_votes > min_grammar_votes:
        return votes == max_votes
    else:
        return votes >= min_grammar_votes

# Mark all unsure words in moderated sentence
def mark_sentence(natural: str, sentence: ModeratedSentence, max_votes: int) -> str:
    words = natural.split(" ")
    pos = 0
    for index in range(len(words)):
        word = words[index]
        while not sentence.tokens[pos].is_lemma():
            pos += 1
        votes = sentence.votes[pos]
        pos += 1 # go to next token
        if votes == 0:
            # red emoji
            words[index] = "🟥" + word
        elif votes < max_votes:
            # orange emoji
            words[index] = "🟧" + word
        else:
            # green emoji
            words[index] = "🟩" + word
    return " ".join(words)
