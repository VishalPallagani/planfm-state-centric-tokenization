"""
Tokenization strategies for Instance Learning Graph (ILG) embeddings.

Provides multiple methods for converting ILGs into fixed-dimensional vectors:
- WLTokenizer: Weisfeiler-Leman color refinement (wraps existing wlplan pipeline)
- SimHashTokenizer: Random projection hashing
- ShortestPathTokenizer: Shortest-path kernel features
- GraphBPETokenizer: Byte-pair encoding on graph structures
- RandomTokenizer: deterministic random baseline embeddings
"""

from code.tokenization.base import TokenizationStrategy
from code.tokenization.multidomain import MultiDomainUnionTokenizer
from code.tokenization.random import RandomTokenizer
from code.tokenization.simhash import SimHashTokenizer
from code.tokenization.shortest_path import ShortestPathTokenizer
from code.tokenization.graphbpe import GraphBPETokenizer

__all__ = [
    "TokenizationStrategy",
    "MultiDomainUnionTokenizer",
    "RandomTokenizer",
    "SimHashTokenizer",
    "ShortestPathTokenizer",
    "GraphBPETokenizer",
]
