# training.py
from dataclasses import dataclass, field
from typing import List

from .config import Configuration
from .context import ContextWindow
from .tokens import TargetToken
from .curriculum import Curriculum, CurriculumStory

# ============================================================
# TrainingSample
# ============================================================

@dataclass
class TrainingSample:
    input_vector: List[float]
    target: TargetToken      # grammar + lemma
    page_index: int          # -1 for terminals

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
        print(f"[{index}] samples = {len(self.samples)}")
