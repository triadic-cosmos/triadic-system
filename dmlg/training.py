# training.py
from dataclasses import dataclass, field
from typing import List

from .config import Configuration
from .context import ContextWindow
from .tokens import TargetToken
from .curriculum import Curriculum, CurriculumSentence

VERBOSE = False

# ============================================================
# TrainingSample
# ============================================================

@dataclass
class TrainingSample:
    input_vector: List[float]
    target: TargetToken      # grammar + lemma
    page_index: int          # -1 for terminals
    position: int

# ============================================================
# TrainingSampleSentence
# ============================================================

@dataclass
class TrainingSampleSentence:
    input_vector: List[float]
    targets: List[TargetToken]
    target_vector: List[float]

# ============================================================
# TrainingBatch
# ============================================================

@dataclass
class TrainingBatch:
    samples: List[TrainingSample] = field(default_factory=list)
    
    def has_samples(self) -> bool:
        return len(self.samples) > 0
    
    def append(self, batch: "TrainingBatch"):
        self.samples += batch.samples
    
    def show(self, index: int):
        if VERBOSE:
            print(f"[{index}] samples = {len(self.samples)}")
