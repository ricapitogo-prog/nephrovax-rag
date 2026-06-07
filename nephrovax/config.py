"""Configuration for the NephroVax RAG system.

Single source of truth for model choices, retrieval parameters, and paths.
All other modules import from here. Change a value here and everywhere
downstream picks it up.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
GUIDELINES_DIR = REPO_ROOT / "guidelines"
EVALS_DIR = REPO_ROOT / "evals"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Embedding: voyage-3 beat voyage-3-large on our 24-case evaluation (top-3: 88% vs 71%)
# See evaluation notes in docs/evaluation.md for details.
EMBEDDING_MODEL = "voyage-3"

# Generation: opus is the strongest Claude model; haiku is much cheaper.
# For production demos use opus; for development/iteration use haiku.
GENERATION_MODEL = "claude-opus-4-5"

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

# Default top-K for single-leaf queries. Generation sees these chunks.
DEFAULT_K = 5

# Higher K for multi-document queries (e.g., "complete vaccination plan for new patient").
# Used when the question is detected as multi-vaccine.
MULTI_CHUNK_K = 10

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

MAX_GENERATION_TOKENS = 1024
