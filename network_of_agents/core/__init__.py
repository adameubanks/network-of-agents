"""
Core mathematical functions for the network of agents simulation.
"""

from .mathematics import (
    update_opinions_pure_degroot,
    create_connected_degroot_network
)

__all__ = [
    "update_opinions_pure_degroot",
    "create_connected_degroot_network"
]