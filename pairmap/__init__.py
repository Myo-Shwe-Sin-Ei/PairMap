"""PairMap — an ontology alignment system for IN3067/INM713 Part 2."""

from .evaluation import evaluate
from .pipeline import run_pipeline
from .serialise import write_alignment

__all__ = ["evaluate", "run_pipeline", "write_alignment"]
