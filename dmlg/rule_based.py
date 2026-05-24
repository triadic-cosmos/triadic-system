from dataclasses import dataclass
from typing import List

from .tokens import Token, TokenLogit
from .context import ContextWindow, ModelInput

@dataclass
class RuleBasedFilter:
    def filter_tokens(self, model_input: ModelInput, tokens: List[Token]) -> List[Token]:
        return [token for token in tokens if self.filter_token(model_input, token)]

    def filter_logits(self, model_input: ModelInput, logits: List[TokenLogit]) -> List[TokenLogit]:
        return [logit for logit in logits if self.filter_token(model_input, logit.token)]

    def filter_token(self, model_input: ModelInput, token: Token) -> bool:
        window: ContextWindow = model_input.window
        last: str = window.last_token().text
        forelast: str = window.forelast_token().text
        text: str = token.text
        
        # duplicate token
        if text == last:
            return False
        
        # duplicate lemma token
        if text[0] != "<" and (text == forelast or last[0] != "<"):
            return False

        # punctuation after wrong token
        if text in ["<PERIOD>", "<EXCLAMATION>", "<QUESTION>", "<COMMA>"]:
            if forelast in ["<DET>", "<CCONJ>"] or last == "<EOL>" or last == "<COMMA>":
                return False
            else:
                return True

        # PRESENT or PAST without VERB
        if text in ["<PRESENT>", "<PRESENT-1S>", "<PRESENT-3S>", "<PAST>"] and forelast != "<VERB>":
            return False

        # VERB directly after tense or after incompatible grammar tokens
        if text == "<VERB>":
            if last in ["<PRESENT>", "<PRESENT-1S>", "<PRESENT-3S>", "<PAST>"]:
                return False

            if forelast in ["<ADJ>", "<ADP>", "<VERB>", "<DET>"]:
                return False

        # VERB without PRESENT or PAST
        if forelast == "<VERB>" and not text in ["<PRESENT>", "<PRESENT-1S>", "<PRESENT-3S>", "<PAST>"]:
            return False
                
        # PLURAL without NOUN
        if text == "<PLURAL>" and forelast != "<NOUN>":
            return False

        # repeating control tokens that are not valid to repeat
        if text in ["<VERB>", "<NOUN>", "<DET>", "<SCONJ>", "<CCONJ>", "<ADP>"] and forelast == text:
            return False
        
        return True
