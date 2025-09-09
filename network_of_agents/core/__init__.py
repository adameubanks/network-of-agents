"""
Core mathematical functions for the network of agents simulation.
"""

from .mathematics import (
    row_normalize,
    calculate_DN,
    calculate_SN,
    calculate_W,
    update_opinions,
    update_opinions_pure_degroot,
    create_connected_degroot_network,
    calculate_S_hat,
    update_edges
)

__all__ = [
    "row_normalize",
    "calculate_DN", 
    "calculate_SN",
    "calculate_W",
    "update_opinions",
    "update_opinions_pure_degroot",
    "create_connected_degroot_network",
    "calculate_S_hat",
    "update_edges"
] 