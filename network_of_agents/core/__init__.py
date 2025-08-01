"""
Core mathematical framework for the network of agents simulation.
"""

from .mathematics import (
    row_normalize,
    calculate_DN,
    calculate_SN,
    calculate_W,
    update_opinions,
    calculate_S_hat,
    update_edges,
)

__all__ = [
    "row_normalize",
    "calculate_DN", 
    "calculate_SN",
    "calculate_W",
    "update_opinions",
    "calculate_S_hat",
    "update_edges",
] 