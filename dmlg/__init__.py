"""
DMLG - Dynamic Modular Language Graph
GLPG - Grammar Lemma Paged Graph
Part of the Triadic System (Triadic Cosmos ecosystem)

This package exposes the public API of the DMLG and GLPG engine.
Internal modules remain accessible but are not exported by default.
"""

# --- Tokens --------------------------------------------------------------

from .tokens import (
    Token,
    TokenPage,
    TokenDictionary,
    TokenLogit,
)

# --- Grammar & Semantics -------------------------------------------------

from .grammar import GrammarEngine
from .semantic import SemanticEngine

# --- Configuration --------------------------------------------------------

from .config import Configuration

# --- Context & Input Encoding --------------------------------------------

from .context import (
    ContextWindow,
    ModelInput,
    InputEncoder,
)

# --- Writer System --------------------------------------------------------

from .writer_agent import WriterAgent

from .writer_story import (
    WriterParams,
    WriterSentence,
    WriterStory
)

from .writer_environment import WriterEnvironment

# --- Multi-Agent System ---------------------------------------------------

from .agent_builder import (
    AgentBuilder,
    TrainingBatchBuilder,
    DATA_FOLDER
)

# --- Curriculum -----------------------------------------------------------

from .curriculum import (
    Curriculum,
    CurriculumSentence
)

# --- Training -------------------------------------------------------------

from .training import (
    TrainingSample,
    TrainingBatch
)

# --- Bias Network -----------------------------------------------------

from .bias import BiasMLP, AMLPBias

# --- Neural / Network -----------------------------------------------------

from .neural import NeuralNetwork
from .glp_network import (
    GlpNetwork,
    TrainingBatch,
)

# --- Public API -----------------------------------------------------------

__all__ = [
    # Tokens
    "Token",
    "TokenPage",
    "TokenDictionary",
    "TokenMapping",
    "TokenLogit",

    # Grammar & Semantics
    "GrammarEngine",
    "SemanticEngine",

    # Configuration
    "Configuration",

    # Context
    "ContextWindow",
    "ModelInput",
    "InputEncoder",

    # Writer System
    "WriterAgent",
    "WriterStory",
    "WriterSentence",
    "WriterEnvironment",

    # Agent Builder
    "AgentBuilder",

    # Curriculum
    "Curriculum",
    "CurriculumStory",
    "CurriculumSentence",

    # Bias
    "BiasMLP",
    "AMLPBias"

    # Neural / Network
    "NeuralNetwork",
    "GlpNetwork",
    "TrainingBatch",
]
