# transition_map.py
from dataclasses import dataclass, field
from typing import Dict, Set, List

from .tokens import Token

@dataclass
class TransitionMap:
    transitions: Dict[str, Set[int]] = field(default_factory=dict)
    token_map: Dict[int, Token] = field(default_factory=dict)

    def learn(self, prev: str, seq: Token):
        self.token_map[seq.id] = seq       
        self.transitions.setdefault(prev, set()).add(seq.id)
        
    def get(self, prev: str) -> List[Token]:
        ids = self.transitions.get(prev)
        if not ids:
            return []
        return [self.token_map[i] for i in ids]
