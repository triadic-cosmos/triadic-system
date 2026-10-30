# training.py
from dataclasses import dataclass, field
from typing import List

from .context import ModelInput
from .tokens import TargetToken

# ============================================================
# TrainingSampleSentence
# ============================================================

@dataclass
class TrainingSampleSentence:
    model_input: ModelInput
    target_tokens: List[TargetToken]
