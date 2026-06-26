"""
DMLG - Dynamic Modular Language Graph
Part of the Triadic System (Triadic Cosmos ecosystem)

This package exposes the public API of the DMLG engine.
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

# --- Sentence Encoding ----------------------------------------------------

from .sentence_encoder import (
    SentenceEncoder,
    EncodedSentence,
    EMPTY_SENTENCE,
)

# --- Writer System --------------------------------------------------------

from .writer_agent import (
    WriterAgent,
    WriterStory,
    WriterSentence,
)

from .writer_environment import WriterEnvironment

# --- Multi-Agent System ---------------------------------------------------

from .agent_builder import (
    AgentBuilder,
    DATA_FOLDER
)

# --- Curriculum -----------------------------------------------------------

from .curriculum import (
    Curriculum,
    CurriculumStory,
    CurriculumSentence,
)

# --- Neural / Network -----------------------------------------------------

from .neural import NeuralNetwork
from .paged_network import (
    PagedNetwork,
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

    # Sentence Encoding
    "SentenceEncoder",
    "EncodedSentence",
    "EMPTY_SENTENCE",

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

    # Neural / Network
    "NeuralNetwork",
    "PagedNetwork",
    "TrainingBatch",
]
