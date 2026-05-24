# grammar_engine.py

from typing import List
import re
import spacy
import inflect
import pyinflect
import language_tool_python

from .tokens import Token
from .config import Configuration

def count_tokens(tokens: List[Token], text: str) -> int:
    count = 0
    for token in tokens:
        if token.text == text:
            count += 1
    return count                

class GrammarEngine:
    """
    Full canonical grammar engine:
    - natural → canonical conversion
    - canonical → natural realization
    - basic grammar validation    
    - grammar fixing
    """

    # -----------------------------
    # Engines
    # -----------------------------

    def __init__(self, configuration: Configuration):
        self.inflect = inflect.engine()
        self.nlp = spacy.load("en_core_web_sm")
        self.fix_tool = language_tool_python.LanguageTool("en-US")
        self.singular_overrides = {"data", "media", "criteria", "bacteria", "phenomena"}
        self.configuration = configuration

    # ============================================================
    # Morphology helpers
    # ============================================================

    def get_plural(self, lemma: str):
        # fix for spacy making verb plural noun
        if lemma == "be":
            return lemma
        return self.inflect.plural(lemma)

    def conjugate_ing(self, verb: str):
        doc = self.nlp(verb)
        token = doc[0]
        form = token._.inflect("VBG")
        return form if form else verb + "ing"

    def conjugate_perfect(self, verb: str):
        # Special case: "be"
        if verb == "be":
            return "been"

        doc = self.nlp(verb)
        token = doc[0]

        # VBN = past participle
        form = token._.inflect("VBN")
        if form:
            return form

        # Fallback: naive regular past participle
        if verb.endswith("e"):
            return verb + "d"
        return verb + "ed"

    def conjugate_present(self, verb: str, person_tag: str):
        # Special case: "be"
        if verb == "be":
            if person_tag == "<1S>": return "am"
            if person_tag == "<3S>": return "is"
            return "are"

        doc = self.nlp(verb)
        token = doc[0]

        if person_tag == "<3S>":
            form = token._.inflect("VBZ")
            return form if form else verb
        else:
            form = token._.inflect("VBP")
            return form if form else verb

    def conjugate_past(self, verb: str):
        # fallback cases
        if verb == "be":
            return "was"
        if verb == "warp":
            return "warped"

        # Use spaCy + pyinflect
        doc = self.nlp(verb)
        token = doc[0]

        # Try irregular/regular past via VBD
        form = token._.inflect("VBD")
        if form:
            return form

        # Fallback: naive regular past
        # (pyinflect sometimes fails for rare or invented verbs)
        if verb.endswith("e"):
            return verb + "d"
        if verb.endswith("y") and verb[-2] not in "aeiou":
            return verb[:-1] + "ied"
        return verb + "ed"

    # ============================================================
    # Grammar validation
    # ============================================================

    # very basic grammatical check using only control tokens
    def basic_validate_grammar_tokens(self, tokens: List[Token]) -> bool:
        tokenlen = len(tokens)
        if tokenlen < 3:
            return False
        if tokens[tokenlen - 1].text != "<EOL>":
            return False
        if tokens[tokenlen - 2].text not in ["<PERIOD>", "<EXCLAMATION>", "<QUESTION>"]:
            return False
        text3 = tokens[tokenlen - 3].text
        if text3[0] == '<' and text3 not in ["<PLURAL>", "<PRESENT>", "<PRESENT-1S>", "<PRESENT-3S>", "<PAST>"]:
            return False

        if count_tokens(tokens, "<VERB>") == 0:
            return False

        return True

    # ============================================================
    # Natural → canonical
    # ============================================================

    def is_verb(self, token):
        if token.pos_ in ("VERB", "AUX"):
            return True
        if token.pos_ == "NOUN" and token.head.pos_ == "NOUN" and token.text.endswith("s"):
            return True
        return False

    def get_tense(self, token):
        tag = token.tag_
        if tag == "VBD":
            if token.text == token.lemma_:
                return "<PRESENT>"
            return "<PAST>"
        if tag in ("VBP", "VBZ"):
            return "<PRESENT>"
        if tag == "VBG":
            return "<ING>"
        if tag == "VBN":
            return "<PERFECT>"
        if tag == "VB":
            return "<BASE>"
        return "<PRESENT>"

    def get_person(self, token):
        # AUX → person from head
        if token.pos_ == "AUX" and token != token.head:
            return self.get_person(token.head)

        subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
        if not subjects:
            return "<3S>"

        subj = subjects[0]
        text = subj.text.lower()

        if text in self.singular_overrides:
            return "<3S>"

        if text == "i": return "<1S>"
        if text == "you": return "<2S>"
        if text in ("he", "she", "it"): return "<3S>"
        if text == "we": return "<1P>"
        if text == "they": return "<3P>"
            
        if "Number=Plur" in subj.morph:
            return "<3P>"
        return "<3S>"

    def merge_tense_person(self, verb: str, tense: str, person: str):
        if tense == "<PRESENT>":
            # 3rd person singular always needs special tag
            if person == "<3S>":
                return "<PRESENT-3S>"
            # 1st person singular only matters for "be"
            if verb == "be" and person == "<1S>":
                return "<PRESENT-1S>"
        return tense

    def convert_token(self, token):
        if token.pos_ == "SPACE":
            return ""

        if token.pos_ == "PUNCT":
            if token.text == "!":
                return "<EXCLAMATION><SPLIT>"
            elif token.text == "?":
                return "<QUESTION><SPLIT>"
            elif token.text == ",":
                return "<COMMA>"
            else:
                return "<PERIOD><SPLIT>"

        # Spacy bugs
        if token.text == "survivors":
            return "<NOUN> survivor <PLURAL>"

        if self.is_verb(token):
            tense = self.get_tense(token)
            if self.noverb and tense == "<PRESENT>":
                # fallback for nouns detected as a verb
                self.noverb = False
                return f"<NOUN> {token.lemma_} <PLURAL>"
            if tense == "<PERFECT>" and token.head.idx <= token.idx:
                tense = "<PAST>"
            if tense in ("<BASE>", "<PERFECT>"):
                return f"{tense} {token.lemma_}"
            elif tense == "<ING>":
                if self.afterverb or self.afteradpadv:
                    tense = "<INGV>"
                else:
                    tense = "<ING>"
                return f"{tense} {token.lemma_}"
            else:
                self.afterverb = True
                person: str = self.get_person(token)
                combined: str = self.merge_tense_person(token.lemma_, tense, person)
                converted = f"<VERB> {token.lemma_} {combined}"
                return converted

        self.afterverb = False
        self.afteradpadv = token.pos_ in ["ADP", "ADV"]
        self.noverb = token.pos_ in ["DET", "ADJ"]

        if token.pos_ == "NOUN":
            converted = f"<NOUN> {token.lemma_}"
            if token.morph.get("Number") == ["Plur"]:
                converted += " <PLURAL>"
            return converted
        
        if token.pos_ == "PRON":
            text = token.text.lower()
            if text in ["its", "his", "her", "their"]:
                return f"<PRONA> {text}"
            else:
                return f"<PRON> {text}"
            
        # This can come from abbreviations
        if token.text == ".":
            return ""

        return f"<{token.pos_}> {token.text.lower()}"

    # fixes some typical issues from spacy
    def fix_canonical(self, canonical: str) -> str:
        fixed = canonical \
                .replace("<NOUN> be <PLURAL> <PRON> you", "<VERB> be <PRESENT> <PRON> you") \
                .replace("<NOUN> can <PLURAL> <PRON> you", "<VERB> can <PRESENT> <PRON> you")
        return fixed

    def convert_to_canonical(self, sentence):
        self.noverb = True
        self.afterverb = False
        self.afteradpadv = False
        doc = self.nlp(sentence)
        parts = []
        for token in doc:
            m = self.convert_token(token)
            if m:
                parts.append(m)
        canonical = " ".join(parts)
        fixed = self.fix_canonical(canonical)
        return fixed

    def convert_to_canonical_tokens(self, sentence) -> List[Token]:
        canonical = self.convert_to_canonical(sentence).replace("<SPLIT>", " <EOL>")
        tokens = [Token(t) for t in canonical.split(" ") if t != ""]
        return tokens

    # ============================================================
    # Canonical → natural
    # ============================================================

    def convert_from_canonical(self, canonical:str) -> str:
        return self.convert_from_canonical_parts(canonical.replace("<SPLIT>", "").split())

    def convert_from_canonical_tokens(self, tokens: List[Token]) -> str:
        parts = [t.text for t in tokens]
        return self.convert_from_canonical_parts(parts)        

    # fixes some typical issues within generated canonical form
    def fix_natural(self, natural: str) -> str:
        return natural \
            .replace("I is", "I am") \
            .replace("I are", "I am") \
            .replace("I can you ", "I can ") \
            .replace("I do you ", "I do ") \
            .replace("I am you ", "I am ")
    
    def convert_from_canonical_parts(self, tokens: List[str]) -> str:
        output = []
        i = 0

        def is_tag(tok):
            return tok.startswith("<") and tok.endswith(">")

        while i < len(tokens):
            tok = tokens[i]

            if tok == "<EOL>":
                i += 1
                continue
            
            # Simple tags
            if tok in ("<DET>", "<NUM>", "<ADJ>", "<ADV>", "<SCONJ>", "<CCONJ>", "<X>",
                       "<PROPN>", "<ADP>", "<PART>", "<BASE>", "<PRON>", "<PRONA>", "<INTJ>"):
                t = tokens[i+1]
                if t == "i":                    
                    output.append("I")
                else:
                    output.append(t)
                i += 2
                continue

            # Nouns
            if tok == "<NOUN>":
                lemma = tokens[i+1]
                if i+2 < len(tokens) and tokens[i+2] == "<PLURAL>":
                    output.append(self.get_plural(lemma))
                    i += 3
                else:
                    output.append(lemma)
                    i += 2
                continue

            # ING
            if tok.startswith("<ING"):
                lemma = tokens[i+1]
                output.append(self.conjugate_ing(lemma))
                i += 2
                continue
            
            # PERFECT
            if tok == "<PERFECT>":
                lemma = tokens[i+1]
                output.append(self.conjugate_perfect(lemma))
                i += 2
                continue

            # Verbs
            if tok == "<VERB>":
                lemma = tokens[i+1]
                tense = tokens[i+2]
                i += 3

                if tense == "<PAST>":
                    verb = self.conjugate_past(lemma)
                elif tense == "<PRESENT>":                    
                    verb = self.conjugate_present(lemma, "<3P>")
                elif tense == "<PRESENT-3S>":
                    verb = self.conjugate_present(lemma, "<3S>")
                elif tense == "<PRESENT-1S>":
                    verb = self.conjugate_present(lemma, "<1S>")                    
                else:
                    verb = lemma

                output.append(verb)
                continue

            # Period
            if tok == "<PERIOD>":
                output.append(".")
                i += 1
                continue

            # Comma
            if tok == "<COMMA>":
                output.append(",")
                i += 1
                continue

            # Exclamation
            if tok == "<EXCLAMATION>":
                output.append("!")
                i += 1
                continue

            # Question
            if tok == "<QUESTION>":
                output.append("?")
                i += 1
                continue

            # Default
            output.append(tok)
            i += 1

        text = " ".join(output)
        text = text.replace(" .", ".").replace(" !", "!").replace(" ?", "?").replace(" ,", ",")
        if text:
            text = text[0].upper() + text[1:]
        fixed = self.fix_natural(text)
        return fixed

    # ============================================================
    # Grammar fixing
    # ============================================================

    def fix_grammar(self, sentence: str) -> str:
        return self.fix_tool.correct(sentence)
