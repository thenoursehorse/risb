"""Solvers for the embedding impurity problem in RISB."""

from .dummy import EmbeddingDummy
from .embedding_atom_diag import EmbeddingAtomDiag

__all__ = ["EmbeddingAtomDiag", "EmbeddingDummy"]
