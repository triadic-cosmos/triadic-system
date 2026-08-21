# rulebased.py
from dataclasses import dataclass, field

from .context import ModelInput
from .tokens import (
    Token,
    ALL_PUNCTIATION_TOKENS,
    END_PUNCTIATION_TOKENS
)

# ============================================================
# TOKEN RULES DATA
# ============================================================

BAD_START = { "them", "him", "her", "whom", "whose", "and", "or", "nor", "but", "yet", "so" }
BAD_END = { "a", "an", "the", "as", "and", "or", "nor", "but", "so", "of", "to", "in", "on", "at", "by", "he", "her", "they", "their" }

NO_PUNCTUATION_TOKENS = {
    "<DET>",
    "<SCONJ>",
    "<CCONJ>",
    "<PART>"
}

CONJUGATION_TOKENS = {
    "<VERB-PRESENT>",
    "<VERB-PRESENT-3S>",
    "<VERB-PRESENT-1S>",
    "<VERB-PAST>"    
}

NOUN_TOKENS = {
    "<NOUN>",
    "<NOUN-PLURAL>",
    "<PROPN>"
}

IGNORED_GRAMMAR = [ "<EOL>", "<X>", "<INTJ>" ]
BAD_START_GRAMMAR = [ "<VERB-PRESENT-1S>", "<VERB-PRESENT-3S>", "<PART>" ]

GRAMMAR_BLACKLIST = {
    # --- VERBS ---
    "<VERB-PRESENT>": { "<PRON>", "<VERB-PRESENT-1S>", "<VERB-PRESENT-3S>" },
    "<VERB-PRESENT-1S>": { "<PRON>", "<VERB-PRESENT-1S>", "<VERB-PRESENT-3S>" },    
    "<VERB-PRESENT-3S>": { "<PRON>", "<VERB-PRESENT-1S>", "<VERB-PRESENT-3S>" },
    "<VERB-PAST>": { "<PRON>", "<VERB-PAST>" },
    "<VERB-ING>": { "<PART>" },
    "<VERB-INGV>": { "<PART>" },
    "<VERB-PERFECT>": { "<PART>" },

    # --- NOUNS ---
    "<NOUN>": { "<NOUN>", "<NOUN-PLURAL>", "<DET>", "<NUM>", "<ADJ>" },
    "<NOUN-PLURAL>": { "<NOUN>", "<NOUN-PLURAL>", "<DET>", "<NUM>" },

    # --- PRONOUNS ---
    "<PRON>": { "<PRON>", "<PRONA>", "<DET>", "<PART>" },
    "<PRONA>": { "<PRON>", "<PRONA>", "<DET>", "<PART>" },

    # --- DETERMINERS ---
    "<DET>": {
        "<DET>", "<PRON>", "<PART>", "<ADP>", "<SCONJ>", "<CCONJ>",
        "<VERB-PRESENT>", "<VERB-PRESENT-1S>", "<VERB-PRESENT-3S>", "<VERB-INGV>",
    },

    # --- ADJECTIVES ---
    "<ADJ>": {
        "<VERB-PRESENT>", "<VERB-PRESENT-1S>", "<VERB-PRESENT-3S>", "<VERB-INGV>",
        "<DET>", "<PRON>", "<PRONA>", "<PART>" },

    # --- ADVERBS ---
    "<ADV>": { "<PART>" },

    # --- ADPOSITIONS ---
    "<ADP>": {
        "<ADP>", "<PART>", "<SCONJ>", "<CCONJ>", 
        "<VERB-PRESENT>", "<VERB-PRESENT-1S>", "<VERB-PRESENT-3S>", "<VERB-PAST>", 
    },

    # --- PARTICLES ---
    "<PART>": { "<PART>" },

    # --- SUBORDINATING CONJUNCTIONS ---
    "<SCONJ>": { "<SCONJ>", "<CCONJ>", "<PART>" },

    # --- COORDINATING CONJUNCTIONS ---
    "<CCONJ>": { "<CCONJ>", "<SCONJ>", "<PART>" },

    # --- NUMBERS ---
    "<NUM>": { "<NUM>", "<PART>" },

    # --- PROPER NOUNS ---
    "<PROPN>": { "<PROPN>", "<NOUN>", "<NOUN-PLURAL>" },

    # --- INTERJECTIONS ---
    "<INTJ>": { "<INTJ>", "<PART>" },

    # --- SPECIAL ---
    "<X>": { "<X>", "<PART>" },
}

# ============================================================
# RULE BASED FILTER : Improve generation using rules ☺
# ============================================================

@dataclass
class RuleBasedFilter:
    def determine_incompatible_lemma(self, model_input: ModelInput) -> set:        
        incompatible = set()

        last = model_input.window.last_token()
        if last.is_eol():
            incompatible.update(BAD_START)        
        elif last.is_lemma():
            incompatible.add(last.lower_text)
                        
            current_tokens = model_input.window.current_tokens

            # repeat with intermediate word (A B A)
            if len(current_tokens) >= 4:
                incompatible.add(current_tokens[-3].lower_text)

            # repeat with intermediate word (A B A B)            
            if len(current_tokens) >= 6:
                cut = current_tokens[-5:]
                if cut[0].lower_text == cut[4].lower_text:
                    incompatible.add(cut[2].lower_text)
            
            # trigram repeat (A B C A B C)
            if len(current_tokens) >= 10:
                cut = current_tokens[-9:]
                if cut[0].lower_text == cut[6].lower_text and \
                   cut[2].lower_text == cut[8].lower_text:
                    incompatible.add(cut[4].lower_text)                
        
        return incompatible
            
    def determine_incompatible_grammar(self, model_input: ModelInput, min_tokens: int) -> set:        
        incompatible = set()
        
        # add all grammar tokens that should be ignored during generation
        incompatible.update(IGNORED_GRAMMAR)

        last = model_input.window.last_token()
        if last.is_grammar():        
            # 1. Prevent repeating the same terminal token
            incompatible.add(last.text) 

            # 2. Punctuation rules
            if last.is_terminal():
                incompatible.update(ALL_PUNCTIATION_TOKENS)
                
            # 3. Bad grammar at start
            if last.is_eol():
                incompatible.update(BAD_START_GRAMMAR)
        else:
            current_tokens = model_input.window.current_tokens
            forelast = model_input.window.forelast_token()
            
            # TODO add all blacklisted grammar tokens
            incompatible.update(GRAMMAR_BLACKLIST[forelast.text])
            
            if last.lower_text in BAD_END or forelast.text in NO_PUNCTUATION_TOKENS:
                incompatible.update(ALL_PUNCTIATION_TOKENS)
            elif len(current_tokens) < min_tokens or \
                not any(t.text.startswith("<VERB-") for t in current_tokens):
                incompatible.update(END_PUNCTIATION_TOKENS)
                
            if forelast.text in CONJUGATION_TOKENS:
                incompatible.update(CONJUGATION_TOKENS)
            elif forelast.text in NOUN_TOKENS:
                incompatible.update(NOUN_TOKENS)

        return incompatible
